### L1-5a144a8ab9  (L1, 2025-04-07, sha 5a144a8ab985, PR #5147)
TITLE: Fix run time error in ROCm platform (#5147)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.routing.router_py; router.py fused elementwise kernel fixes ROCm runtime errors.
ARTIFACT_HINTS: L1.routing.router_py
FILES: python/sglang/srt/layers/moe/router.py (+7/-1); python/sglang/srt/layers/elementwise.py (+15/-2); python/sglang/srt/layers/quantization/fp8_utils.py (+1/-0)
BODY: Obsolete: #5124  ⏎  ⏎ ## Motivation ⏎  ⏎ ``` ⏎  ⏎ ``` ⏎  ⏎ When running the latest docker image "lmsysorg/sglang:v0.4.4.post4-rocm630", there are some errors happened. ⏎ **Error 1** ⏎ ``` ⏎ File "/sgl-workspace/sglang/python/sglang/srt/models/grok.py", line 320, in forward ⏎     fused_rmsnorm( ⏎   File "/sgl-workspace/sglang/python/sglang/srt/layers/elementwise.py", line 260, in fused_rmsnorm ⏎     fused_rmsnorm_kernel[(bs,)]( ⏎   File "/usr/local/lib/python3.12/dist-packages/triton/runtime/jit.py", line 330, in <lambda> ⏎     return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs) ⏎                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎   File "/usr/local/lib/python3.12/dist-packages/triton/runtime/jit.py", line 653, in run ⏎     kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata, ⏎   File "/usr/local/lib/python3.12/dist-packages/triton/backends/amd/driver.py", line 479, in __call__ ⏎     self.launch(*args, **kwargs) ⏎ RuntimeError: Triton Error [HIP]:  Code: 1, Messsage: invalid argument ⏎  ⏎ ``` ⏎  ⏎ **Error 2** ⏎ ``` ⏎ File "/sgl-workspace/sglang/python/sglang/srt/layers/quantization/fp8_utils.py", line 324, in apply_fp8_linear ⏎     output = torch._scaled_mm( ⏎              ^^^^^^^^^^^^^^^^^ ⏎ RuntimeError: false INTERNAL ASSERT FAILED at "/app/pytorch/aten/src/ATen/hip/HIPDataType.h":102, please report a bug to PyTorch. Cannot convert ScalarType Float8_e4m3fn to hipDataType. ⏎ ``` ⏎  ⏎ ## Modifications ⏎  ⏎ 1) change the triton call launch minimum warp size ⏎ 2) add torch.float8_e4m3fnuz support in input_to_float8 function ⏎  ⏎ ## Checklist

### L1-a73c4df438  (L1, 2025-04-08, sha a73c4df4387a, PR #5150)
TITLE: Add optimized native kernels in sgl-kernel (#5150)
SOURCES: path_core, symbol_pickaxe
STAGE1: introduce; artifacts=L1.hardware.cpu_npu_musa; Adds optimized CPU MoE and top-k native kernels.
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: sgl-kernel/csrc/cpu/moe.cpp (+1247/-0); sgl-kernel/csrc/cpu/moe_int8.cpp (+830/-0); sgl-kernel/csrc/cpu/topk.cpp (+406/-0); sgl-kernel/csrc/cpu/activation.cpp (+79/-0); sgl-kernel/csrc/cpu/bmm.cpp (+122/-0); sgl-kernel/csrc/cpu/common.h (+164/-0); sgl-kernel/csrc/cpu/decode.cpp (+1119/-0); sgl-kernel/csrc/cpu/extend.cpp (+621/-0); sgl-kernel/csrc/cpu/gemm.cpp (+507/-0); sgl-kernel/csrc/cpu/gemm.h (+130/-0); sgl-kernel/csrc/cpu/gemm_int8.cpp (+489/-0); sgl-kernel/csrc/cpu/interface.cpp (+120/-0); sgl-kernel/csrc/cpu/norm.cpp (+221/-0); sgl-kernel/csrc/cpu/qkv_proj.cpp (+504/-0); sgl-kernel/csrc/cpu/rope.cpp (+129/-0); sgl-kernel/csrc/cpu/shm.cpp (+659/-0); sgl-kernel/csrc/cpu/shm.h (+11/-0); sgl-kernel/csrc/cpu/torch_extension_cpu.cpp (+224/-0); sgl-kernel/csrc/cpu/vec.h (+115/-0); sgl-kernel/setup_cpu.py (+95/-0)
LABELS: high priority, sgl-kernel, intel, cpu
PERF_LINES: Prefill. latency: 2.79226 s, throughput:    366.73 token/s | Decode.  latency: 0.07376 s, throughput:     13.56 token/s | Decode.  latency: 0.06361 s, throughput:     15.72 token/s | Decode.  latency: 0.06163 s, throughput:     16.23 token/s | Decode.  latency: 0.06307 s, throughput:     15.86 token/s | Decode.  latency: 0.05998 s, throughput:     16.67 token/s
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This pull request is a follow up on https://github.com/sgl-project/sglang/issues/2807 to enable and optimize sglang performance on CPU devices. In this patch, optimized C++ kernels are provided including: ⏎ * activations ⏎ * layernorms ⏎ * gemm (bfloat16, int8) ⏎ * extend attention (bfloat16) ⏎ * decode attention (bfloat16) ⏎ * allreduce and allgather ⏎ * moe (bfloat16, int8) ⏎ * rope ⏎  ⏎ Specifically, we are are targeting at optimizing DeepSeek R1 617B on CPU devices. And right now the performance on Xeon6 with single batch size and 1024 input tokens and 1024 output tokens are: ⏎ ``` ⏎ torch profiler chrome trace saved to Trace_prefill_DS-R1-INT8-TP6-BS1-1024-1024_20250402_batch1_input1024_output1024.trace.json.gz ⏎ Prefill. latency: 2.79226 s, throughput:    366.73 token/s ⏎ Decode.  latency: 0.07376 s, throughput:     13.56 token/s ⏎ Decode.  latency: 0.06361 s, throughput:     15.72 token/s ⏎ Decode.  latency: 0.06163 s, throughput:     16.23 token/s ⏎ Decode.  latency: 0.06307 s, throughput:     15.86 token/s ⏎ Decode.  latency: 0.05998 s, throughput:     16.67 token/s ⏎ ``` ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ This PR contains changes in C++ parts from `sglang/sgl-kernel/csrc/cpu` and we decide to upstream the C++ kernels first so as not to make the PR overwhelming. We will upstream changes to sglang python layers one by one later on. ⏎  ⏎ Also please notice that the CPU build now relies on `setup_cpu.py`. Originally we made modifications on `setup.py` but I just found out this file has been removed. ⏎ ## Checklist

### L1-bc3f6db2dd  (L1, 2025-04-08, sha bc3f6db2dd6a, PR #5068)
TITLE: [Fix] DeepEP Compatibility with Low Latency (#5068)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: repair_correctness; artifacts=L1.ep.deepep_dispatcher; DeepEP dispatcher buffers made compatible between normal and low-latency modes.
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+145/-118); python/sglang/srt/model_executor/forward_batch_info.py (+1/-1); python/sglang/srt/models/deepseek_v2.py (+1/-1); python/sglang/srt/server_args.py (+1/-0)
LABELS: high priority
PERF_LINES: As title, make DeepEP normal buffer compatible with low_latency buffer, simply tested both intra-node and inter-node can run successfully. | | Type | DeepEP Auto | DeepEP Normal (disable CUDA Graph)| DeepEP Low Latency | | | MoE Version | Concurrency | Input | Output | Num Requests | Input Throughput(tok/s) | Output Throughput (tok/s) | Total Throughput (tok/s) | | | MoE Version | Concurrency | In
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ As title, make DeepEP normal buffer compatible with low_latency buffer, simply tested both intra-node and inter-node can run successfully. ⏎  ⏎ Support matrix ⏎ | Type | DeepEP Auto | DeepEP Normal (disable CUDA Graph)| DeepEP Low Latency | ⏎ |------------------|------------------|------------------|------------------| ⏎ | Intra-node      | ✓         | ✓           | ✓                | ⏎ | Inter-node      | ✓         | ✓           | ✓                | ⏎  ⏎ **performance (1 node, auto mode, H20-3e)** ⏎ - single node (H20-3e) ⏎ ```sh ⏎ # DeepEP MoE (auto) ⏎ python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --trust-remote-code \ ⏎ --tp 8 --host 0.0.0.0 --port 30000 --enable-deepep-moe --deepep-mode auto \ ⏎ --max-running-requests 128 --disable-radix-cache --mem-fraction-static 0.9 --stream-output \ ⏎ --cuda-graph-max-bs 128 ⏎ ``` ⏎ | MoE Version | Concurrency | Input | Output | Num Requests | Input Throughput(tok/s) | Output Throughput (tok/s) | Total Throughput (tok/s) | ⏎ | -- | -- | -- | -- | -- | -- | -- | -- | ⏎ | DeepEP origin | 127.97 | 1000 | 1000 | 512 | 581.94 | 581.94 | 1163.87 | ⏎ | DeepEP auto | 127.95 | 1000 | 1000 | 512 | 954.56  | 954.56  | **1909.13** | ⏎ | EPMoE | 127.94 | 1000 | 1000 | 512 | 862.52  | 862.52  | 1725.04  | ⏎  ⏎ **performance (2 nodes, auto mode, H20-3e)** ⏎ ```sh ⏎ # node 0 ⏎ python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --trust-remote-code \ ⏎ --tp 16 --dist-init-addr 10.6.131.5:5000 --nnodes 2 --node-rank 0 --enable-deepep-moe \ ⏎ --max-running-requests 128 --disable-radix-cache --mem-fraction-static 0.9 --stream-output \ ⏎ --cuda-graph-max-bs 128 ⏎  ⏎ # node 1 ⏎ python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --trust-remote-code \ ⏎ --tp 16 --dist-init-addr 10.6.131.5:5000 --nnodes 2 --node-rank 1 --enable-deepep-moe \ ⏎ --max-running-requests 128 --disable-radix-cache --mem-fraction-static 0.9 --stream-output \ ⏎ --cuda-graph-max-bs 128 ⏎  ⏎ # bench ⏎ python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompt 512 \ ⏎ --random-input 1000 --random-output 1000 --random-range-ratio 1 --host 127.0.0.1 --port 30000 \ ⏎ --max-concurrency 128 ⏎ ``` ⏎ | MoE Version | Concurrency | Input | Output | Num Reqs | Input (tok/s) | Output (tok/s) | Total (tok/s) | Mean TTFT (ms) | Mean ITL (ms) | ⏎ | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | ⏎ | Pure TP 16 | 127.94 | 1000 | 1000 | 512 | 1139.01 | 1139.01 | 2278.02 | 9086.90 | 103.34 | ⏎ | DeepEP auto | 127.94 | 1000 | 1000 | 512 | **1146.25**  | **1146.25**  | **2292.50** | **8164.64** | **103.56** | ⏎ | EPMoE | 127. …[truncated]

### L1-90caf06c00  (L1, 2025-04-08, sha 90caf06c0064, PR #5180)
TITLE: fix: use DeepEPDispatcher on CUDA (#5180)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L1.ep.deepep_dispatcher; DeepSeek model selects DeepEPDispatcher on CUDA for DeepEP mode.
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+2/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-76c48a0913  (L1, 2025-04-08, sha 76c48a0913b9, PR #5179)
TITLE: [DeepEP] fix: import buffer error (#5179)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_build_dependency; artifacts=L1.ep.deepep_dispatcher; DeepEP token_dispatcher import path for buffer is fixed.
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+2/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-f730362ee2  (L1, 2025-04-09, sha f730362ee207, PR #5086)
TITLE: reduce moe_align_block_size_kernel small batch mode overhead (#5086)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: optimize; artifacts=L1.align.cuda_aot; moe_align CUDA kernel reduces small-batch overhead.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.align.cuda_aot
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+1/-1); sgl-kernel/csrc/moe/moe_align_kernel.cu (+111/-44); sgl-kernel/benchmark/bench_moe_align_block_size.py (+31/-10); sgl-kernel/tests/test_moe_align.py (+0/-1)
PERF_LINES: 100%|████████████████████████████████████████████████████████████████████████| 1319/1319 [01:25<00:00, 15.36it/s] | Latency: 90.543 s | Output throughput: 1532.037 token/s
BODY: ## Motivation ⏎  ⏎  ⏎ ## Acc test ⏎  ⏎ I set `token_cnts_buffer` and `cumsum_buffer` to `torch.empty` in `fused_moe.py`: ⏎  ⏎ ```python ⏎ token_cnts_buffer = torch.empty( ⏎             (num_experts + 1) * num_experts, ⏎             dtype=torch.int32, ⏎             device=topk_ids.device, ⏎         ) ⏎         cumsum_buffer = torch.empty( ⏎             num_experts + 1, dtype=torch.int32, device=topk_ids.device ⏎         ) ⏎ ``` ⏎  ⏎ Acc result: ⏎  ⏎ ```shell ⏎ ➜  sglang python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --num-shots 8 --port 30001 ⏎  ⏎ 100%|████████████████████████████████████████████████████████████████████████| 1319/1319 [01:25<00:00, 15.36it/s] ⏎ Accuracy: 0.952 ⏎ Invalid: 0.000 ⏎ Latency: 90.543 s ⏎ Output throughput: 1532.037 token/s ⏎ ``` ⏎  ⏎ ## Kernel unit-test ⏎  ⏎ ![图片](https://github.com/user-attachments/assets/b8cd621e-dd61-41d6-8058-20ecf472327f) ⏎  ⏎  ⏎ ## Benchmark In H200 ⏎  ⏎ main branch: ⏎  ⏎ ```shell ⏎ 📊 Running performance benchmark for 8 experts... ⏎ moe-align-block-size-performance: ⏎      num_tokens  num_experts  topk        SGL       Triton        VLLM ⏎ 0           1.0          8.0   1.0  18.975999    67.359999   14.624000 ⏎ 1           1.0          8.0   2.0  19.136000    23.264000   14.607999 ⏎ 2           1.0          8.0   4.0  20.384001    63.519999   14.624000 ⏎ 3           1.0          8.0   8.0  19.424001    62.368002   14.656000 ⏎ 4           1.0         32.0   1.0  20.128001    62.912002   16.640000 ⏎ 5           1.0         32.0   2.0  21.536000    63.263997   16.640000 ⏎ 6           1.0         32.0   4.0  21.632001    69.023997   16.576000 ⏎ 7           1.0         32.0   8.0  20.256000    59.712000   16.640000 ⏎ 8           1.0         64.0   1.0  22.816001    56.912001   20.128001 ⏎ 9           1.0         64.0   2.0  22.816001    28.672000   20.256000 ⏎ 10          1.0         64.0   4.0  22.784000    69.440000   20.223999 ⏎ 11          1.0         64.0   8.0  22.816001    65.024003   20.288000 ⏎ 12          1.0        128.0   1.0  24.224000    64.511999   30.975999 ⏎ 13          1.0        128.0   2.0  23.040000    68.335995   30.944001 ⏎ 14          1.0        128.0   4.0  24.288001    63.167997   30.944001 ⏎ 15          1.0        128.0   8.0  23.135999    63.808002   31.040000 ⏎ 16          1.0        256.0   1.0  26.559999    58.240000   65.471999 ⏎ 17          1.0        256.0   2.0  26.591999    70.528001   65.471999 ⏎ 18          1.0        256.0   4.0  26.815999    61.471999   65.632001 ⏎ 19          1.0        256.0   8.0  27.872000    59.680000   65.664001 ⏎ 20          8.0          8.0   1.0  19.200001    65.568000   14.6880 …[truncated]

### L1-4065248214  (L1, 2025-04-09, sha 406524821457, PR #5194)
TITLE: Support Llama4 fp8 inference (#5194)
SOURCES: path_core, symbol_pickaxe
STAGE1: extend_support; artifacts=L1.triton.fused_moe; fused_moe supports Llama4 FP8 quantized inference path.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+33/-18); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py (+4/-0); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+66/-45); python/sglang/srt/layers/quantization/fp8_utils.py (+9/-0); python/sglang/srt/layers/quantization/w8a8_fp8.py (+154/-4); python/sglang/srt/layers/quantization/w8a8_int8.py (+1/-0); python/sglang/srt/model_loader/loader.py (+10/-3); python/sglang/srt/model_loader/weight_utils.py (+4/-1); python/sglang/srt/models/deepseek_v2.py (+24/-16); python/sglang/srt/models/llama4.py (+1/-1); python/sglang/srt/models/mllama4.py (+52/-18); test/srt/run_suite.py (+1/-0); test/srt/test_int8_kernel.py (+1/-0); test/srt/test_triton_moe_channel_fp8_kernel.py (+177/-0)
LABELS: high priority, quant
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Support llama4 fp8 inference for [meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 ⏎ ](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8) @zhyncs @ispobock @zhaochenyang20  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Benchmark  ⏎  ⏎ ``` ⏎ # launch ⏎ python -m sglang.launch_server --model-path meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 --tp 8 ⏎  ⏎ # gsm8k and mmlu ⏎ python benchmark/gsm8k/bench_sglang.py --num-questions 1400 ⏎ python benchmark/mmlu/bench_sglang.py ⏎ ``` ⏎ | Dataset | Score | ⏎ |-----|------| ⏎ | gsm8k | 93.6 | ⏎ | mmlu | 86.2 | ⏎  ⏎  ⏎ ## TODO ⏎  ⏎  ⏎ ## Checklist

### L1-60bcbf2a35  (L1, 2025-04-11, sha 60bcbf2a35e2, PR #5298)
TITLE: remove moe_align_block_size torch.zeros in small batch/expert mode (#5298)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: optimize; artifacts=L1.align.cuda_aot,L1.triton.fused_moe; fused_moe stops zeroing moe_align small-batch buffers.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ Follow [pr 5086](https://github.com/sgl-project/sglang/pull/5086) ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-adca585bfb  (L1, 2025-04-13, sha adca585bfb59, PR #5277)
TITLE: [DeepEP] Reduce routed scaling overhead (#5277)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: optimize; artifacts=L1.ep.deepep_dispatcher; DeepEP low-latency path uses masked_scale to reduce routed-scaling overhead.
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+9/-10)
LABELS: high priority
PERF_LINES: In the `--deepep-mode=low_latency mode`, `final_hidden_states` is a tensor with the shape `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hid | Applying `routed_scaling_factor` to the entire `final_hidden_states` tooks 350 us: | Using `masked_scale` tooks 36.5 us.
BODY: ## Motivation ⏎  ⏎ In the `--deepep-mode=low_latency mode`, `final_hidden_states` is a tensor with the shape `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]`, and most of its values are masked. Applying `routed_scaling_factor` to the entire tensor would result in substantial memory access overhead. The introduction of `masked_scale` avoids scaling the masked portions, thereby reducing memory access and lowering latency. ⏎  ⏎  ⏎ On H20 running Deepseek-V3-5layers, with EP4, 60 concurrency: ⏎  ⏎ Applying `routed_scaling_factor` to the entire `final_hidden_states` tooks 350 us: ⏎ ![image](https://github.com/user-attachments/assets/ceb1b0ea-3570-4e5b-b542-a5dc8b6ef255) ⏎  ⏎ Using `masked_scale` tooks 36.5 us. ⏎ ![image](https://github.com/user-attachments/assets/5d0a060b-eaf9-4c1b-987b-f1e197be2810) ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-38076dea84  (L1, 2025-04-14, sha 38076dea8425, PR #5371)
TITLE: apply fused moe gate in ds v3/r1 (#5371)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: change_default; artifacts=L1.routing.topk_py,L1.routing.fused_gate; topk.py applies fused MoE gate kernel for DeepSeek V3/R1.
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+37/-16)
PERF_LINES: 100%|█████████████████████████████████████████████████████████████████████████| 1319/1319 [00:54<00:00, 24.24it/s] | Latency: 54.740 s | Output throughput: 2529.836 token/s | |qps|Input token throughput (tok/s)|Output token throughput (tok/s)|Total token throughput (tok/s)| | - qps=4: 5.9%+ | - qps=8: 5.5%+ | - qps=16: 8.1%+
BODY: ## torch profile ⏎  ⏎ ```shell ⏎ python3 -m sglang.bench_serving --backend sglang --num-prompts 2 --request-rate 1 --port 30001 --flush-cache --warmup-requests 1 --profile ⏎ ``` ⏎  ⏎ ### main branch ⏎  ⏎ ![图片](https://github.com/user-attachments/assets/6e5d87ca-1090-47da-b05e-1e1124583048) ⏎  ⏎ ### pr ⏎  ⏎ ![图片](https://github.com/user-attachments/assets/e09ede25-c6c6-4481-8428-30d053c96e4e) ⏎  ⏎ Only one kernel now. ⏎  ⏎ 36us->8us. ⏎  ⏎  ⏎ ## fused_moe_gate gsm8k acc test ⏎  ⏎ ```shell ⏎ SGL_ENABLE_JIT_DEEPGEMM=0 python3 -m sglang.launch_server --model /DeepSeek-V3 --tp 8 --trust-remote-code --port 30001 ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --num-shots 8  --port 30001 ⏎  ⏎ 100%|█████████████████████████████████████████████████████████████████████████| 1319/1319 [00:54<00:00, 24.24it/s] ⏎ Accuracy: 0.950 ⏎ Invalid: 0.000 ⏎ Latency: 54.740 s ⏎ Output throughput: 2529.836 token/s ⏎ ``` ⏎  ⏎ ## fused moe gate performance(in H200) ⏎  ⏎ ```shell ⏎ SGL_ENABLE_JIT_DEEPGEMM=0 python3 -m sglang.launch_server --model /DeepSeek-V3 --tp 8 --trust-remote-code --port 30001 ⏎ python3 -m sglang.bench_serving --backend sglang --num-prompts 300 --request-rate 1 --port 30001 --flush-cache --warmup-requests 20 ⏎ ``` ⏎  ⏎ |qps|Input token throughput (tok/s)|Output token throughput (tok/s)|Total token throughput (tok/s)| ⏎ |---|---|---|---| ⏎ |4(main)| 719.99| 456.44| 1176.43| ⏎ |4(pr)  | 763.11| 483.77| 1246.88| ⏎ |8(main)| 840.96| 533.13| 1374.09| ⏎ |8(pr)  | 887.35| 562.54| 1449.89| ⏎ |16(main)| 892.55| 565.83| 1458.38| ⏎ |16(pr)  | 964.91| 611.70| 1576.61| ⏎  ⏎  ⏎ - qps=4: 5.9%+ ⏎ - qps=8: 5.5%+ ⏎ - qps=16: 8.1%+

### L1-8e09b37077  (L1, 2025-04-17, sha 8e09b370777c, PR #5440)
TITLE: Sgl kernel fused_moe_gate support n_shared_experts (#5440)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: extend_support; artifacts=L1.routing.fused_gate; fused_moe_gate kernel and wrapper add n_shared_experts support.
ARTIFACT_HINTS: L1.routing.fused_gate
FILES: sgl-kernel/csrc/common_extension.cc (+2/-1); sgl-kernel/csrc/moe/moe_fused_gate.cu (+81/-28); sgl-kernel/include/sgl_kernel_ops.h (+8/-2); sgl-kernel/python/sgl_kernel/moe.py (+18/-2); sgl-kernel/tests/test_moe_fused_gate.py (+31/-5)
PERF_LINES: 100%|███████████████████████████████████████████████████████| 1319/1319 [00:56<00:00, 23.30it/s] | Latency: 58.947 s | Output throughput: 2358.942 token/s
BODY: ## Motivation ⏎  ⏎  ⏎ ```shell ⏎ python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code --port 30001 ⏎  ⏎ ➜  sglang python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --num-shots 8 --port 30001 ⏎ 100%|███████████████████████████████████████████████████████| 1319/1319 [00:56<00:00, 23.30it/s] ⏎ Accuracy: 0.955 ⏎ Invalid: 0.000 ⏎ Latency: 58.947 s ⏎ Output throughput: 2358.942 token/s ⏎ ```  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-bed05878f6  (L1, 2025-04-18, sha bed05878f6c8, PR #5461)
TITLE: fix kimi vl running bug after rebase main (#5461)
SOURCES: path_core, symbol_pickaxe
STAGE1: repair_correctness; artifacts=L1.routing.topk_py; topk.py fixes Kimi VL MoE runtime shape/index error.
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+2/-0)
DEEP_STUDY: deep-study correctness case sglang:bed05878f6: class=shape_alignment_edge; symptom=crash_or_exception; introducing=unknown
BODY: ## Motivation ⏎  ⏎ Fix: ⏎  ⏎ ```shell ⏎ File "/eightT/open_source/sglang/python/sglang/srt/models/deepseek_v2.py", line 1177, in forward_normal ⏎     hidden_states = self.mlp(hidden_states) ⏎                     ^^^^^^^^^^^^^^^^^^^^^^^ ⏎   File "/eightT/open_source/sglang/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl ⏎     return self._call_impl(*args, **kwargs) ⏎            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎   File "/eightT/open_source/sglang/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl ⏎     return forward_call(*args, **kwargs) ⏎            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎   File "/eightT/open_source/sglang/python/sglang/srt/models/deepseek_v2.py", line 277, in forward ⏎     return self.forward_normal(hidden_states) ⏎            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎   File "/eightT/open_source/sglang/python/sglang/srt/models/deepseek_v2.py", line 286, in forward_normal ⏎     self.experts(hidden_states=hidden_states, router_logits=router_logits) ⏎   File "/eightT/open_source/sglang/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl ⏎     return self._call_impl(*args, **kwargs) ⏎            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎   File "/eightT/open_source/sglang/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl ⏎     return forward_call(*args, **kwargs) ⏎            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎   File "/eightT/open_source/sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py", line 627, in forward ⏎     final_hidden_states = self.quant_method.apply( ⏎                           ^^^^^^^^^^^^^^^^^^^^^^^^ ⏎   File "/eightT/open_source/sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py", line 135, in apply ⏎     return self.forward( ⏎            ^^^^^^^^^^^^^ ⏎   File "/eightT/open_source/sglang/python/sglang/srt/custom_op.py", line 18, in forward ⏎     return self._forward_method(*args, **kwargs) ⏎            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎   File "/eightT/open_source/sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py", line 169, in forward_cuda ⏎     topk_weights, topk_ids = select_experts( ⏎                              ^^^^^^^^^^^^^^^ ⏎   File "/eightT/open_source/sglang/python/sglang/srt/layers/moe/topk.py", line 293, in select_experts ⏎     topk_weights, topk_ids = biased_grouped_topk( ⏎                              ^^^^^^^^^^^^^^^^^^^^ ⏎   File "/eightT/open_source/sglang/python/sglang/srt/layers/moe/topk.py", line 236, in biased_grouped_topk ⏎     return moe_fused_gate( ⏎          …[truncated]

### L1-1e0806f30b  (L1, 2025-04-18, sha 1e0806f30b99, PR #5340)
TITLE: Fix DeepGEMM masked cannot be run on groups not being multiple or 4 (#5340)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.ep.layer,L1.runner.deep_gemm; EP DeepGEMM path removes invalid group-multiple assertion.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+0/-3)
BODY: ## 2025.04.17 ⏎  ⏎ Originally I added some assertions to ensure we do not accidentally enter the slow branch of deepgemm preparation, but @ch-wan reviewed and suggested to remove it, thus the new code does not contain this. ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-d58e354472  (L1, 2025-04-19, sha d58e35447203, PR #5504)
TITLE: simplify the control logic for using shared experts fusion (#5504)
SOURCES: path_core, symbol_pickaxe
STAGE1: change_default; artifacts=L1.routing.topk_py,L1.triton.fused_moe,L1.ep.layer; Shared-experts fusion control changes default flag semantics and MoE path selection.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.routing.topk_py, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+3/-0); python/sglang/srt/layers/moe/fused_moe_native.py (+4/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+2/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+7/-0); python/sglang/srt/layers/moe/topk.py (+15/-10); python/sglang/srt/layers/quantization/__init__.py (+1/-0); python/sglang/srt/layers/quantization/blockwise_int8.py (+2/-0); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+4/-0); python/sglang/srt/layers/quantization/fp8.py (+2/-0); python/sglang/srt/layers/quantization/moe_wna16.py (+2/-0); python/sglang/srt/layers/quantization/w8a8_fp8.py (+2/-0); python/sglang/srt/layers/quantization/w8a8_int8.py (+2/-0); python/sglang/srt/managers/schedule_batch.py (+0/-1); python/sglang/srt/model_executor/model_runner.py (+0/-1); python/sglang/srt/models/deepseek_v2.py (+21/-31); python/sglang/srt/server_args.py (+2/-11)
BODY: ## Motivation ⏎  ⏎ Follow [pr 5440](https://github.com/sgl-project/sglang/pull/5440) , and following  @merrymercy ‘s suggestion, simplify the control logic for using shared experts fusion. Setting `--n-shared-experts-fusion=0` by default means it is not enabled, while other values indicate it is enabled. In the parameters, it is noted that setting it to the current `--tp-size` can achieve the best performance. However, this optimization lacks a tuning config for fused MOE in a wide range of scenarios. Therefore, the control over this optimization is given to the user instead of being enabled by default." ⏎  ⏎ Merge [pr 5440](https://github.com/sgl-project/sglang/pull/5440) and release the new sgl-kernel before merging this PR. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-d9dd529854  (L1, 2025-04-20, sha d9dd529854f7, PR #5571)
TITLE: enable DeepSeek V3 shared_experts_fusion in sm90 (#5571)
SOURCES: symbol_pickaxe
STAGE1: change_default; artifacts=L1.routing.topk_py,L1.triton.fused_moe; DeepSeek model enables shared_experts_fusion automatically on SM90.
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+12/-0)
PERF_LINES: **The difference in the data set of 5000-1000 was due to fluctuations. I retested it once, and there was no difference compared to before the fusion. The other  | 100%|███████████████████████████████████████████████████████████████████████████████████| 50/50 [01:42<00:00,  2.06s/it] | Request throughput (req/s):              0.49 | Input token throughput (tok/s):          486.29 | Output token thr
BODY: ## H200 Benchmark ⏎  ⏎ I tested the benchmark using the command provided by https://github.com/sgl-project/sglang/issues/5514. To avoid warmup, I turned off deepgemm. Below are the results and detailed test records. ⏎  ⏎ <img width="676" alt="图片" src="https://github.com/user-attachments/assets/617075f5-aea8-432a-b4e0-da33995ab775" /> ⏎  ⏎ **The difference in the data set of 5000-1000 was due to fluctuations. I retested it once, and there was no difference compared to before the fusion. The other cases all showed acceleration. The acceleration ratios were 4% (1000-2000), 4.8% (10k-500), and 1.1% (30k-100) respectively** . ⏎  ⏎ The reason for enabling only in SM>=90 is that the fused moe kernel currently only has tuning config for SM>=90. It is more optimal not to fuse in cases where there is no tuning config available, as the routed experts section mostly have tuning. ⏎  ⏎ ```shell ⏎ # launch server ⏎ # SGLang uses FA3 backend by default since v0.4.5.post1 ⏎ # Use dp 8 for offline use case ⏎ SGL_ENABLE_JIT_DEEPGEMM=0 python3 -m sglang.launch_server --model /DeepSeek-V3 --tp 8 --trust-remote-code --enable-dp-attention --dp-size 8 ⏎  ⏎ # Random 1k, 2k ⏎ python3 -m sglang.bench_serving --backend sglang-oai --num-prompts 50 --request-rate 10 --dataset-name random --random-input-len 1000 --random-output-len 2000 --random-range-ratio 1 ⏎  ⏎ # Random 5k, 1k ⏎ python3 -m sglang.bench_serving --backend sglang-oai --num-prompts 50 --request-rate 10 --dataset-name random --random-input-len 5000 --random-output-len 1000 --random-range-ratio 1 ⏎  ⏎ # Random 10k, 500 ⏎ python3 -m sglang.bench_serving --backend sglang-oai --num-prompts 50 --request-rate 10 --dataset-name random --random-input-len 10000 --random-output-len 500 --random-range-ratio 1 ⏎  ⏎ # Random 30k, 100 ⏎ python3 -m sglang.bench_serving --backend sglang-oai --num-prompts 50 --request-rate 10 --dataset-name random --random-input-len 30000 --random-output-len 100 --random-range-ratio 1 ⏎ ``` ⏎  ⏎ - main: ⏎  ⏎ ```shell ⏎ benchmark_args=Namespace(backend='sglang-oai', base_url=None, host='0.0.0.0', port=None, dataset_name='random', dataset_path='', model=None, tokenizer=None, num_prompts=50, sharegpt_output_len=None, sharegpt_context_len=None, random_input_len=1000, random_output_len=2000, random_range_ratio=1.0, request_rate=10.0, max_concurrency=None, output_file=None, disable_tqdm=False, disable_stream=False, return_logprob=False, seed=1, disable_ignore_eos=False, extra_request_body=None, apply_chat_template=False, profile=False, lora_name=None, prompt_suffix='', pd_seperated=False, flush_cache=False, warmup_requests=1, gsp_num_groups …[truncated]

### L1-463d4b7400  (L1, 2025-04-20, sha 463d4b7400e4, PR #5567)
TITLE: Fix DeepEP cannot run on latest master (#5567)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: repair_correctness; artifacts=L1.ep.layer; EP MoE layer fix restores DeepEP on latest master.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+2/-0)
BODY: ## Motivation ⏎  ⏎ waiting for CI ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-e62c49557d  (L1, 2025-04-22, sha e62c49557dfc, PR #5281)
TITLE: [1/2] Add FP8 Blockscale MoE CUTLASS kernel for Blackwell (#5281)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: introduce; artifacts=L1.cutlass.fp8_blockwise; Adds FP8 blockscale CUTLASS MoE kernel and Python wrapper.
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise
FILES: sgl-kernel/CMakeLists.txt (+1/-0); sgl-kernel/csrc/common_extension.cc (+5/-0); sgl-kernel/csrc/moe/cutlass_moe_helper.cu (+142/-0); sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu (+386/-0); sgl-kernel/include/sgl_kernel_ops.h (+14/-0); sgl-kernel/python/sgl_kernel/__init__.py (+6/-1); sgl-kernel/python/sgl_kernel/moe.py (+30/-0); sgl-kernel/tests/test_fp8_blockwise_moe.py (+148/-0)
LABELS: high priority
BODY: ## Motivation ⏎ Functionality integration for FP8 blockscale MoE CUTLASS kernels on Blackwell.  ⏎ Huge thanks to @depaulmillz for CUTLASS library support.  ⏎   ⏎ cc @kushanam  ⏎  ⏎ ## Modifications  ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-711efe7814  (L1, 2025-04-23, sha 711efe781426, PR #5435)
TITLE: Integrating PD disaggregation with DP attention and DeepEP (#5435)
SOURCES: subject_keyword, release_notes
STAGE1: extend_support; artifacts=L1.ep.deepep_dispatcher; PD disaggregation integration adds DP attention plus DeepEP support.
ARTIFACT_HINTS: -
FILES: python/sglang/srt/disaggregation/decode.py (+46/-5); python/sglang/srt/disaggregation/prefill.py (+16/-0); python/sglang/srt/managers/data_parallel_controller.py (+10/-3)
LABELS: high priority
PERF_LINES: MOONCAKE_CONFIG_PATH=./pd_node2.json python -m sglang.launch_server --model-path /dev/shm/DeepSeek-V3-0324 --disaggregation-mode decode --host 10.10.38.2 --port | MOONCAKE_CONFIG_PATH=./pd_node3.json python -m sglang.launch_server --model-path /dev/shm/DeepSeek-V3-0324 --disaggregation-mode decode --host 10.10.38.3 --port | - Accuracy: 93.5% ~ 95.0%
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Support DP attention and DeepEP in PD disaggregation. Discussed with @ByronHsu @ShangmingCai @whybeyoung. Tested by @liz-badada. ⏎  ⏎  ⏎ ## Evaluation ⏎  ⏎  ⏎  ⏎ - Prepare configuration files ⏎  ⏎ ```txt ⏎ # pd_node0.json ⏎ { ⏎     "local_hostname": "10.10.37.16", ⏎     "metadata_server": "http://10.10.37.16:8998/metadata", ⏎     "protocol": "rdma", ⏎     "device_name": "mlx5_7" ⏎ } ⏎  ⏎ # pd_node1.json ⏎ { ⏎     "local_hostname": "10.10.38.1", ⏎     "metadata_server": "http://10.10.37.16:8998/metadata", ⏎     "protocol": "rdma", ⏎     "device_name": "mlx5_1" ⏎ } ⏎  ⏎ # pd_node2.json ⏎ { ⏎     "local_hostname": "10.10.38.2", ⏎     "metadata_server": "http://10.10.37.16:8998/metadata", ⏎     "protocol": "rdma", ⏎     "device_name": "mlx5_1" ⏎ } ⏎  ⏎ # pd_node3.json ⏎ { ⏎     "local_hostname": "10.10.38.3", ⏎     "metadata_server": "http://10.10.37.16:8998/metadata", ⏎     "protocol": "rdma", ⏎     "device_name": "mlx5_1" ⏎ } ⏎  ⏎ ``` ⏎  ⏎ - Launch servers ⏎ ```bash ⏎ MOONCAKE_CONFIG_PATH=./pd_node0.json python -m sglang.launch_server --model-path /dev/shm/DeepSeek-V3-0324 --disaggregation-mode prefill --host 10.10.37.16 --port 30000 --trust-remote-code --dist-init-addr 10.10.37.16:5000 --nnodes 2 --node-rank 0 --tp-size 16 --dp-size 8 --enable-dp-attention --enable-deepep-moe --deepep-mode normal --mem-fraction-static 0.8 ⏎  ⏎ MOONCAKE_CONFIG_PATH=./pd_node1.json python -m sglang.launch_server --model-path /dev/shm/DeepSeek-V3-0324 --disaggregation-mode prefill --host 10.10.38.1 --port 30000 --trust-remote-code --dist-init-addr 10.10.37.16:5000 --nnodes 2 --node-rank 1 --tp-size 16 --dp-size 8 --enable-dp-attention --enable-deepep-moe --deepep-mode normal --mem-fraction-static 0.8 ⏎  ⏎ MOONCAKE_CONFIG_PATH=./pd_node2.json python -m sglang.launch_server --model-path /dev/shm/DeepSeek-V3-0324 --disaggregation-mode decode --host 10.10.38.2 --port 30001 --trust-remote-code --dist-init-addr 10.10.38.2:5000 --nnodes 2 --node-rank 0 --tp-size 16 --dp-size 8 --enable-dp-attention --enable-deepep-moe --deepep-mode low_latency --mem-fraction-static 0.8 --cuda-graph-max-bs 128 --max-running-requests 128 ⏎  ⏎ MOONCAKE_CONFIG_PATH=./pd_node3.json python -m sglang.launch_server --model-path /dev/shm/DeepSeek-V3-0324 --disaggregation-mode decode --host 10.10.38.3 --port 30001 --trust-remote-code --dist-init-addr 10.10.38.2:5000 --nnodes 2 --node-rank 1 --tp-size 16 --dp-size 8 --enable-dp-attention --enable-deepep-moe --deepep-mode low_latency --mem-fraction-static 0.8 --cuda-graph-max-bs 128 --max-running-requests 128 ⏎ ``` ⏎  ⏎ - Launch proxy ⏎  ⏎ ```bash ⏎ python3 -m sglang.srt.disaggregation.mini_lb -- …[truncated]

### L1-a086a11305  (L1, 2025-04-26, sha a086a113050f, PR #4971)
TITLE: Use sgl-kernel sgl_per_token_group_quant_int8 (#4971)
SOURCES: path_core
STAGE1: integrate; artifacts=L1.triton.fused_moe; fused_moe switches INT8 per-token group quantization to sgl-kernel op.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+7/-1); python/sglang/srt/layers/quantization/int8_kernel.py (+32/-1)
BODY: ## Motivation ⏎ Based on https://github.com/sgl-project/sglang/pull/4396, use the `sgl_per_token_group_quant_int8` method in the new version of sgl-kernel. ⏎  ⏎ ## Modifications ⏎ Modify `int8_kernel.py` and `fused_moe.py`, and add `sglang_per_token_group_quant_int8` method and related calls. ⏎  ⏎ ## Checklist

### L1-d364b9b0f2  (L1, 2025-04-28, sha d364b9b0f26d, PR #5816)
TITLE: ROCm: update AITER (#5816)
SOURCES: path_core
STAGE1: repair_build_dependency; artifacts=L1.triton.fused_moe; ROCm AITER update changes fused_moe_triton AITER usage/imports.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+2/-2); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+15/-17); .github/workflows/pr-test-amd.yml (+6/-6); 3rdparty/amd/tuning/benchmark_moe_rocm.py (+1/-1); docker/Dockerfile.rocm (+2/-2); python/sglang/srt/layers/quantization/fp8.py (+20/-22); python/sglang/srt/layers/quantization/fp8_utils.py (+2/-2)
BODY: `co-author: kkHuang-amd` ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎ AITER update to v0.1.1 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-cc4a80caf6  (L1, 2025-04-29, sha cc4a80caf604, PR #5830)
TITLE: [PD] Fix Assertion failed: /DeepEP/csrc/kernels/internode.cu:483, condition: ibgda_get_state()->num_rc_per_pe >= num_channels #134 (#5830)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: repair_correctness; artifacts=L1.ep.deepep_dispatcher; DeepEP dispatcher channel setting fixes DeepEP assertion failure.
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+1/-3)
ISSUES: #134 Assertion failed: /DeepEP/csrc/kernels/internode.cu:483, condition: ibgda_get_state()->num_rc_per_pe >= num_channels
BODY: ## Motivation ⏎  ⏎ Fix https://github.com/deepseek-ai/DeepEP/issues/134

### L1-acc816d8a2  (L1, 2025-05-08, sha acc816d8a24e, PR #5626)
TITLE:  DeepEP normal support deepgemm-contiguous (#5626)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: extend_support; artifacts=L1.ep.layer,L1.ep.deepep_dispatcher; EP layer/dispatcher kernels add DeepGEMM contiguous mode support.
ARTIFACT_HINTS: L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+340/-2); python/sglang/srt/layers/moe/ep_moe/layer.py (+120/-1); python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+97/-54); python/sglang/srt/layers/quantization/deep_gemm.py (+5/-0); python/sglang/srt/layers/quantization/fp8_kernel.py (+2/-2); python/sglang/srt/models/deepseek_v2.py (+4/-0)
BODY: ## Motivation ⏎ DeepEP normal support deepgemm-contiguous ⏎ The contiguous mode has passed debugging in normal mode, just finished testing on gsm8k, and accuracy is okay.  ⏎ ## Modifications ⏎  ⏎ ## TODO: ⏎ * There are still some details in code style that need adjustment. ⏎ * If fp8 quantization is performed before dispatch, a deep ep buffer error occurs. I will troubleshoot this later. ⏎ * Compatibility testing for auto mode. ⏎ * Performance testing -- Compare with normal base.

### L1-198b9056d1  (L1, 2025-05-14, sha 198b9056d18c, PR #6274)
TITLE: [AMD] Fix Llama 4 Scout and Maverick accuracy issues on MI300X (#6274)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.triton.fused_moe; FusedMoE AITER path fixes apply_router_weight_on_input accuracy for AMD models.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+13/-0)
PERF_LINES: <body lang=en-US style='font-family:Calibri;font-size:11.0pt'> | SGLang (8*MI300X)     (v0.4.6.post2-rocm630 + this PR) | 75.0 | 81.2
BODY: ## Motivation ⏎  ⏎ Fix the following models on AMD GPUs. ⏎ - meta-llama/Llama-4-Scout-17B-16E-Instruct ⏎ - meta-llama/Llama-4-Maverick-17B-128E-Instruct ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ - Fix `FusedMoE `with `SGLANG_AITER_MOE=1` path when `apply_router_weight_on_input=True` ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ### Llama-4-Scout-17B-16E-Instruct ⏎ ``` ⏎ SGLANG_AITER_MOE=1 python -m sglang.launch_server --model-path meta-llama/Llama-4-Scout-17B-16E-Instruct/ --port 30000 --tp 8 --mem-fraction-static 0.8 --context-length 65536 ⏎ lm_eval --model local-chat-completions --model_args model=meta-llama/Llama-4-Scout-17B-16E-Instruct/,base_url=http://localhost:30000/v1/chat/completions,num_concurrent=128,timeout=999999,max_gen_toks=2048 --tasks mmlu_pro --batch_size 128 --apply_chat_template --num_fewshot 0 ⏎ ``` ⏎ ### Llama-4-Maverick-17B-128E-Instruct ⏎ ``` ⏎ SGLANG_AITER_MOE=1 python -m sglang.launch_server --model-path meta-llama/Llama-4-Maverick-17B-128E-Instruct --port 30000 --tp 8 --mem-fraction-static 0.8 --context-length 65536 ⏎ lm_eval --model local-chat-completions --model_args model=meta-llama/Llama-4-Maverick-17B-128E-Instruct,base_url=http://localhost:30000/v1/chat/completions,num_concurrent=128,timeout=999999,max_gen_toks=2048 --tasks mmlu_pro --batch_size 128 --apply_chat_template --num_fewshot 0 ⏎ ``` ⏎  ⏎ <html xmlns:o="urn:schemas-microsoft-com:office:office" ⏎ xmlns:dt="uuid:C2F41010-65B3-11d1-A29F-00AA00C14882" ⏎ xmlns="http://www.w3.org/TR/REC-html40"> ⏎  ⏎ <head> ⏎  ⏎ <meta name=ProgId content=OneNote.File> ⏎ <meta name=Generator content="Microsoft OneNote 15"> ⏎ </head> ⏎  ⏎ <body lang=en-US style='font-family:Calibri;font-size:11.0pt'> ⏎  ⏎  ⏎ <p style='margin:0in;margin-left:.375in;font-family:Calibri;font-size:11.0pt'>Ref: ⏎ <a ⏎ href="https://github.com/sgl-project/sglang/blob/main/docs/references/llama4.md">https://github.com/sgl-project/sglang/blob/main/docs/references/llama4.md</a></p> ⏎  ⏎ <div style='direction:ltr'> ⏎  ⏎  ⏎   | Llama-4-Scout-17B-16E-Instruct | Llama-4-Maverick-17B-128E-Instruct ⏎ -- | -- | -- ⏎ Official Benchmark | 74.3 | 80.5 ⏎ SGLang (8*H100) | 75.2 | 80.7 ⏎ SGLang (8*MI300X)     (v0.4.6.post2-rocm630 + this PR) | 75.0 | 81.2 ⏎  ⏎  ⏎  ⏎ </div> ⏎  ⏎  ⏎ </body> ⏎  ⏎ </html> ⏎  ⏎ This PR also resolves this issue: https://github.com/sgl-project/sglang/issues/5402. ⏎  ⏎ CC: @HaiShaw

### L1-f194e14fb7  (L1, 2025-05-15, sha f194e14fb7ff, PR #6147)
TITLE: Reduce MoE memory usage (#6147)
SOURCES: path_core, symbol_pickaxe
STAGE1: optimize; artifacts=L1.ep.layer; EP MoE kernels/layer reduce MoE memory usage.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+10/-2); python/sglang/srt/layers/moe/ep_moe/layer.py (+58/-35); python/sglang/srt/models/deepseek_v2.py (+3/-3); python/sglang/srt/utils.py (+4/-0)
BODY: ## Motivation ⏎  ⏎ I realized there is a more lightweight approach to do something similar as DisposableTensor, and please refer to the `dispose_tensor` function. ⏎  ⏎ For figures below, note that the aspect ratio is different for each figure, so the height of each box cannot be compared directly. Instead, can only check relative relationships between boxes. ⏎  ⏎ In addition, the "prefill" here means the default deepgemm approach, while the one in #5085 was the triton kernel approach, so the figure will be different. ⏎  ⏎ Prefill (before): ⏎  ⏎ ![image](https://github.com/user-attachments/assets/39373cbb-c4bc-49a6-a2dc-e7ba79cca6b7) ⏎  ⏎ Decode (before): ⏎  ⏎ ![image](https://github.com/user-attachments/assets/27215252-c10e-4934-8825-8a5428c43c67) ⏎  ⏎ Prefill (after): ⏎  ⏎ ![image](https://github.com/user-attachments/assets/1117940b-055f-41c0-9169-20dc26ff8f37) ⏎  ⏎  ⏎ Decode (after): ⏎  ⏎ ![image](https://github.com/user-attachments/assets/377e1bf7-7a4e-42e1-a94b-c0f3fb12822c) ⏎  ⏎ p.s. a future optimization is to optimize the deepep dispatch output tensor, but it is of low priority since it does not take much space ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-6fc9357503  (L1, 2025-05-16, sha 6fc935750336, PR #5694)
TITLE: [2/2] Add python wrapper for CUTLASS FP8 Blockscale MoE Kernel.  (#5694)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:production-kernel-provenance
STAGE1: integrate; artifacts=L1.cutlass.adapters,L1.cutlass.fp8_blockwise; Adds Python adapter wrapping CUTLASS FP8 blockscale MoE kernel.
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+207/-0); python/sglang/srt/layers/quantization/fp8.py (+90/-0); python/sglang/srt/layers/quantization/fp8_utils.py (+6/-0); sgl-kernel/CMakeLists.txt (+1/-0); sgl-kernel/csrc/common_extension.cc (+7/-3); sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu (+111/-36); sgl-kernel/csrc/moe/prepare_moe_input.cu (+128/-0); sgl-kernel/include/sgl_kernel_ops.h (+18/-1); sgl-kernel/python/sgl_kernel/__init__.py (+1/-0); sgl-kernel/python/sgl_kernel/moe.py (+36/-0); python/sglang/test/test_cutlass_moe.py (+278/-0); sgl-kernel/tests/test_fp8_blockwise_moe.py (+12/-0)
LABELS: high priority
PERF_LINES: Using the benchmark we provided in the PR, we have found our fused_expert layer with CUTLASS 4.0 in CUDA graph mode has ~30%-40% speedup over Triton in CUDA gra | Cutlass fused_experts time: 0.106 ms (median) [0.101 - 0.107] | Triton  fused_experts time: 0.148 ms (median) [0.146 - 0.149] | Cutlass fused_experts time: 0.155 ms (median) [0.150 - 0.157] | Triton  fused_experts time: 0.207 ms (median)
BODY: ### NOTE ⏎  ⏎ The current CUTLASS 3.9 in SGLang will experience: 1. Kernel hang 2. Perf slowdown for the this MoE kernel. ⏎ I'll update our CUTLASS dependency in another PR, as it breaks some of the existing sm90 templates.  ⏎  ⏎ ## Motivation ⏎ Using the benchmark we provided in the PR, we have found our fused_expert layer with CUTLASS 4.0 in CUDA graph mode has ~30%-40% speedup over Triton in CUDA graph mode on small batch sizes.  ⏎  ⏎  ⏎ For Deepseek V3/R1 models, where  ⏎ `{'num_experts': 256, 'topk': 8, 'hidden_size': 7168, 'shard_intermediate_size': 512, 'dtype': torch.bfloat16, 'block_shape': [128, 128]}` ⏎  ⏎  The result of `python3 python/sglang/test/test_cutlass_moe.py`: ⏎ ``` ⏎ --- Batch Size: 1 --- ⏎ Config: E=256, topk=8, H=7168, I_shard=512, dtype=torch.bfloat16, block_shape=[128, 128] ⏎ Warming up... ⏎ Benchmarking Cutlass fused_experts... ⏎ Benchmarking Triton fused_experts... ⏎ Cutlass fused_experts time: 0.106 ms (median) [0.101 - 0.107] ⏎ Triton  fused_experts time: 0.148 ms (median) [0.146 - 0.149] ⏎  ⏎ --- Batch Size: 4 --- ⏎ Config: E=256, topk=8, H=7168, I_shard=512, dtype=torch.bfloat16, block_shape=[128, 128] ⏎ Warming up... ⏎ Benchmarking Cutlass fused_experts... ⏎ Benchmarking Triton fused_experts... ⏎ Cutlass fused_experts time: 0.155 ms (median) [0.150 - 0.157] ⏎ Triton  fused_experts time: 0.207 ms (median) [0.205 - 0.207] ⏎  ⏎ --- Batch Size: 8 --- ⏎ Config: E=256, topk=8, H=7168, I_shard=512, dtype=torch.bfloat16, block_shape=[128, 128] ⏎ Warming up... ⏎ Benchmarking Cutlass fused_experts... ⏎ Benchmarking Triton fused_experts... ⏎ Cutlass fused_experts time: 0.234 ms (median) [0.229 - 0.235] ⏎ Triton  fused_experts time: 0.287 ms (median) [0.285 - 0.288] ⏎  ⏎ --- Batch Size: 16 --- ⏎ Config: E=256, topk=8, H=7168, I_shard=512, dtype=torch.bfloat16, block_shape=[128, 128] ⏎ Warming up... ⏎ Benchmarking Cutlass fused_experts... ⏎ Benchmarking Triton fused_experts... ⏎ Cutlass fused_experts time: 0.350 ms (median) [0.345 - 0.351] ⏎ Triton  fused_experts time: 0.372 ms (median) [0.369 - 0.372] ⏎  ⏎ --- Batch Size: 32 --- ⏎ Config: E=256, topk=8, H=7168, I_shard=512, dtype=torch.bfloat16, block_shape=[128, 128] ⏎ Warming up... ⏎ Benchmarking Cutlass fused_experts... ⏎ Benchmarking Triton fused_experts... ⏎ Cutlass fused_experts time: 0.534 ms (median) [0.520 - 0.535] ⏎ Triton  fused_experts time: 0.532 ms (median) [0.527 - 0.534] ⏎  ⏎ --- Batch Size: 64 --- ⏎ Config: E=256, topk=8, H=7168, I_shard=512, dtype=torch.bfloat16, block_shape=[128, 128] ⏎ Warming up... ⏎ Benchmarking Cutlass fused_experts... ⏎ Benchmarking Triton fused_experts... ⏎ Cutlass fused_experts time: 0.677 ms (media …[truncated]

### L1-2716830802  (L1, 2025-05-17, sha 2716830802ae, PR #6175)
TITLE: Speed up when having padding tokens in DeepEP (#6175)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: optimize; artifacts=L1.routing.topk_py; topk.py skips padding-token work to speed DeepEP cases.
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+38/-4); python/sglang/srt/model_executor/cuda_graph_runner.py (+4/-0); python/sglang/srt/model_executor/forward_batch_info.py (+4/-0); python/sglang/srt/models/deepseek_v2.py (+7/-5)
LABELS: high priority
PERF_LINES: PYTHONUNBUFFERED=1 SGLANG_TORCH_PROFILER_DIR=/host_home/temp_sglang_server2local python3 -m sglang.launch_server --model-path /dev/shm/DeepSeek-R1 --trust-remot | * baseline: 6 tok/s/gpu | * PR: 29 tok/s/gpu
BODY: ## Motivation ⏎  ⏎ test ⏎  ⏎ ``` ⏎ PYTHONUNBUFFERED=1 SGLANG_TORCH_PROFILER_DIR=/host_home/temp_sglang_server2local python3 -m sglang.launch_server --model-path /dev/shm/DeepSeek-R1 --trust-remote-code --dist-init-addr 192.168.0.55:5757 --nnodes 2 --node-rank ${MY_NODE_RANK} --tp-size ${num_gpu} --dp-size ${num_gpu} --enable-dp-attention --mem-fraction-static 0.8 --chunked-prefill-size $((128*${num_gpu})) --max-running-requests $((${num_gpu}*128)) --context-length 4096 --disable-radix-cache --enable-deepep-moe --deepep-mode low_latency --cuda-graph-bs 128 --decode-log-interval 1 ⏎  ⏎ python3 -m sglang.bench_one_batch_server --model-path /dev/shm/DeepSeek-R1 --base-url http://localhost:30000 --batch-size 16 --input-len 1 --output-len 2048 --skip-warmup ⏎ ``` ⏎  ⏎ * baseline: 6 tok/s/gpu ⏎ * PR: 29 tok/s/gpu ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-e3b8a72291  (L1, 2025-05-17, sha e3b8a72291af, PR #6348)
TITLE: [fix] illegal memory in _fwd_kernel_ep_scatter_2 and _fwd_kernel_ep_gather (#6348)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.ep.layer; EP scatter/gather kernels switch index width to avoid illegal memory.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+23/-9)
BODY: ## Motivation ⏎  ⏎  ⏎ when deploying large scale ep deepseek, the prefill node sometimes meets illegal memory error with heavy workload as following ⏎ <img width="697" alt="image" src="https://github.com/user-attachments/assets/f4d5061b-59df-47c7-95cd-ede1713cfe77" /> ⏎ this is caused by the index overflow in _fwd_kernel_ep_scatter_2. ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ replace int32 index with int64 to avoid overflow.  ⏎  ⏎ CUDA_LAUNCH_BLOCKING=1 python test.py to reproduce the illegal memory error: ⏎ ``` ⏎ import torch ⏎ from sglang.srt.layers.moe.ep_moe.kernels import ep_scatter, ep_gather ⏎ import time ⏎  ⏎ N, D, D_ = 235234*3, 7168, 56 ⏎ E = 9 ⏎ N_ = 304000*3 ⏎ recv_x = torch.randn((N, D), device="cuda").to(torch.float8_e4m3fn) ⏎ recv_x_scale = torch.randn((N, D_), device="cuda").to(torch.float) ⏎ num_recv_tokens_per_expert = 3*torch.tensor([0, 107904, 178304, 240000, 273536, 284288, 292736, 296832, 301440, 304000], device="cuda").to(torch.int32) ⏎ num_recv_tokens_per_expert[1:] = num_recv_tokens_per_expert[1:] - num_recv_tokens_per_expert[:-1] ⏎ recv_topk = torch.empty((N, E), device="cuda").to(torch.int64) ⏎ recv_topk.fill_(-1) ⏎ recv_topk_weight = torch.empty((N, E), device="cuda").to(torch.float) ⏎ for i in range(E): ⏎     num = num_recv_tokens_per_expert[i + 1].item() ⏎     recv_topk[-num:, i] = i ⏎ expert_start_loc = torch.empty_like(num_recv_tokens_per_expert) ⏎ output_tensor = torch.zeros((N_, D), device="cuda").to(torch.float8_e4m3fn) ⏎ output_tensor_scale = torch.empty((N_, D_), device="cuda").to(torch.float) ⏎ m_indices = torch.empty(N_, device="cuda").to(torch.int32) ⏎ output_index = torch.randn((N, E), device="cuda").to(torch.int64) ⏎  ⏎ ep_scatter( ⏎     recv_x, ⏎     recv_x_scale, ⏎     recv_topk, ⏎     num_recv_tokens_per_expert, ⏎     expert_start_loc, ⏎     output_tensor, ⏎     output_tensor_scale, ⏎     m_indices, ⏎     output_index, ⏎ ) ⏎  ⏎ down_output = torch.randn((N_, D), device="cuda").to(torch.bfloat16) ⏎ gather_out = torch.empty((N, D), device="cuda").to(torch.bfloat16) ⏎  ⏎ ep_gather(down_output, recv_topk, recv_topk_weight, output_index, gather_out) ⏎  ⏎ print(output_tensor[0]) ⏎ print(gather_out[0]) ⏎ print("passed examine") ⏎ ``` ⏎  ⏎ Many thanks to Ch-Wan (<cwan39@gatech.edu>) for identifying the index overflow as the cause and for providing main part of the test script. ⏎ ``` ⏎  ⏎ ``` ⏎ ## Checklist

### L1-fd08c04821  (L1, 2025-05-17, sha fd08c0482129, PR #6257)
TITLE: Support custom DeepEP tuning config (#6257)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: extend_support; artifacts=L1.ep.deepep_dispatcher; DeepEP dispatcher accepts custom tuning configuration.
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+35/-5); python/sglang/srt/model_executor/model_runner.py (+1/-0); python/sglang/srt/server_args.py (+7/-0); python/sglang/srt/managers/schedule_batch.py (+1/-0); python/sglang/srt/utils.py (+7/-0); test/srt/test_moe_deepep.py (+28/-0)
BODY: ## Motivation ⏎  ⏎ Btw that test looks a bit stale, thus a few lines of diff are there to make that test run again ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-c471d39eb9  (L1, 2025-05-19, sha c471d39eb9e2, PR #6386)
TITLE: Support loading weights when physical experts are different from logical experts (#6386)
SOURCES: path_core
STAGE1: extend_support; artifacts=L1.ep.layer; EP layer supports physical experts differing from logical experts.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+23/-0); python/sglang/srt/managers/expert_location.py (+14/-1)
BODY: ## Motivation ⏎  ⏎ subtract diff from https://github.com/sgl-project/sglang/pull/4957 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-e98afbe042  (L1, 2025-05-19, sha e98afbe042cf, PR #6385)
TITLE: Support dispatching logical to physical experts (#6385)
SOURCES: path_core
STAGE1: extend_support; artifacts=L1.routing.topk_py,L1.ep.layer; topk/EP layer add logical-to-physical expert dispatch contract.
ARTIFACT_HINTS: L1.routing.topk_py, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+4/-0); python/sglang/srt/layers/moe/topk.py (+18/-0); python/sglang/srt/managers/expert_distribution.py (+2/-1); python/sglang/srt/managers/expert_location.py (+58/-3); python/sglang/srt/managers/expert_location_dispatch.py (+91/-0); python/sglang/srt/managers/schedule_batch.py (+1/-0); python/sglang/srt/model_executor/model_runner.py (+1/-1); python/sglang/srt/models/deepseek_v2.py (+2/-0); python/sglang/srt/server_args.py (+7/-0)
BODY: ## Motivation ⏎  ⏎ subtract diff from https://github.com/sgl-project/sglang/pull/4957 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-a40aecc5a3  (L1, 2025-05-21, sha a40aecc5a3a5, PR #6468)
TITLE: Fix num_qps_per_rank computation when providing custom DeepEP configuration (#6468)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L1.ep.deepep_dispatcher; DeepEP dispatcher fixes num_qps_per_rank for custom configuration.
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+21/-9)
BODY: ## Motivation ⏎  ⏎ Note: it is also made public, b/c I need to access it in TBO ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-a071dc4084  (L1, 2025-05-21, sha a071dc4084ea, PR #6467)
TITLE: Tiny add stage assertions to DeepEPDispatcher to avoid misuse (#6467)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L1.ep.deepep_dispatcher; DeepEPDispatcher adds stage assertions guarding invalid use.
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+20/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-cfe48c5902  (L1, 2025-05-21, sha cfe48c590228, PR #6419)
TITLE: [CPU] Fix build issue (#6419)
SOURCES: path_core, symbol_pickaxe
STAGE1: repair_build_dependency; artifacts=L1.hardware.cpu_npu_musa; CPU sgl-kernel build/import fixes cover CPU MoE sources.
ARTIFACT_HINTS: -
FILES: sgl-kernel/csrc/cpu/moe.cpp (+9/-9); sgl-kernel/csrc/cpu/CMakeLists.txt (+10/-41); sgl-kernel/csrc/cpu/bmm.cpp (+2/-1); sgl-kernel/csrc/cpu/gemm.cpp (+2/-1); sgl-kernel/csrc/cpu/gemm_fp8.cpp (+1/-1); sgl-kernel/csrc/cpu/gemm_int8.cpp (+2/-2); sgl-kernel/csrc/cpu/interface.cpp (+5/-6); sgl-kernel/csrc/cpu/qkv_proj.cpp (+6/-6); sgl-kernel/csrc/cpu/shm.h (+1/-1); sgl-kernel/csrc/cpu/torch_extension_cpu.cpp (+94/-42); sgl-kernel/pyproject_cpu.toml (+0/-4); sgl-kernel/setup_cpu.py (+4/-2); test/srt/cpu/test_gemm.py (+14/-18); test/srt/cpu/test_shared_expert.py (+7/-9)
LABELS: high priority, intel, cpu
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎  ⏎ 1. Simplify the `CMakeLists.txt` to automatically detect all `.cpp` files under the `csrc/cpu/` directory as `SOURCES`. ⏎ 2. Fix the issue where `sgl_kernel` cannot be imported properly. ⏎  ⏎ The following commands are expected to work correctly after this PR. ⏎  ⏎ ``` ⏎ cd sgl-kernel/ ⏎ cp pyproject_cpu.toml pyproject.toml ⏎ pip install -v . ⏎ python -c "import sgl_kernel" ⏎ ``` ⏎  ⏎ cc @chunyuan-w  ⏎  ⏎ ## Checklist

### L1-e9feb48838  (L1, 2025-05-21, sha e9feb4883830, PR #6308)
TITLE: [RL] Remove the w13 weight_scale and input_scale for UnquantizedEPMoE… (#6308)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.ep.layer; UnquantizedEPMoE removes unreloadable scale fields for RL memory release.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+5/-14)
BODY: …Method ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎ When doing RL training, we may release all the parameters with `/release_memory_occupation` to free the memory occupied by the inference engine, which will also released all the `input_scale`s and `weight_scale`s. ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ The origin `w13_weight_scale` in `UnquantizedEPMoEMethod` does not support reloading (as the shape should be `(num_experts_per_partition, 2)`). And I found that for the `UnquantizedEPMoEMethod`, we don't need to instantiate `w13_weight_scale` and `w13_input_scale`, so removing them could be a better solution than allocating twice the origin memory. ⏎  ⏎ And note that we do need to reload the `w2_input_scale`, because if we set that to `None`, it will be initialized to `torch.ones` during `EpMoE.forward_normal`. So I need to change the condition in `_load_fp8_scale` to allow loading `w2_input_scale` from a random value to 1. ⏎  ⏎ Thank you for your time on reviewing this PR :) ⏎  ⏎ ## Checklist

### L1-3ded6235c9  (L1, 2025-05-23, sha 3ded6235c9e4, PR #6404)
TITLE: Add fp8 fused_experts kernel for CPU in sgl-kernel and add UT (#6404)
SOURCES: path_core, symbol_pickaxe
STAGE1: extend_support; artifacts=L1.hardware.cpu_npu_musa; Adds CPU FP8 fused_experts MoE kernel and tests.
ARTIFACT_HINTS: -
FILES: sgl-kernel/csrc/cpu/moe.cpp (+76/-28); sgl-kernel/csrc/cpu/moe_fp8.cpp (+291/-12); sgl-kernel/csrc/cpu/gemm.h (+26/-0); sgl-kernel/csrc/cpu/torch_extension_cpu.cpp (+4/-1); sgl-kernel/setup_cpu.py (+0/-116); test/srt/cpu/test_moe.py (+259/-0); test/srt/cpu/utils.py (+96/-0)
LABELS: high priority, sgl-kernel, intel, cpu
BODY: ## Motivation ⏎  ⏎  ⏎ This PR is a follow-up on https://github.com/sgl-project/sglang/issues/2807 and https://github.com/sgl-project/sglang/pull/5150 to add **fp8** **fused_experts** kernel for CPU. The bf16 and int8 fused_experts kernel is already added in https://github.com/sgl-project/sglang/pull/5150. ⏎  ⏎ This PR also adds UTs for bf16, int8 and fp8 fused_experts kernels for CPU. ⏎  ⏎ ## Modifications ⏎ The main change is the C++ kernels for fp8 fused_experts on CPU: `sgl-kernel/csrc/cpu/moe_fp8.cpp` ⏎ The UTs for fused_experts OPs on CPU: `test/srt/cpu/test_moe.py` ⏎ Add `gemm_fp8.cpp` and `moe_fp8.cpp` into `sgl-kernel/csrc/cpu/CMakeLists.txt` to use `pyproject_cpu.toml`.

### L1-2f42749184  (L1, 2025-05-23, sha 2f42749184ca, PR #6474)
TITLE: Fix topk inference performance reduce (#6474)
SOURCES: path_core
STAGE1: repair_performance; artifacts=L1.routing.topk_py; topk.py fix removes performance regression from prior logic.
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+2/-0)
BODY: ## Motivation ⏎ When the following logic is added to `topk.py`, the inference performance will be significantly affected: ⏎ https://github.com/sgl-project/sglang/blob/66324895c6925c86c2b8c811ebb6dfb93ae42356/python/sglang/srt/layers/moe/topk.py#L267-L269 ⏎  ⏎ Run command: ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path /path/to/DeepSeek-V3-0324 --trust-remote-code --host 0.0.0.0 --port 30000 --attention-backend flashinfer --n-share-experts-fusion 16 --tp 16 --dist-init-addr IP:20000 --nnodes 2 --node-rank 0 ⏎ ``` ⏎ Ref: https://github.com/sgl-project/sglang/pull/6175 ⏎  ⏎ ## Modifications ⏎ Add `num_token_non_padded` judgment logic. If it is None, directly return the previous result. ⏎  ⏎ ## Checklist

### L1-a564e001b5  (L1, 2025-05-27, sha a564e001b532, PR #6673)
TITLE: Fix DeepEP error in Qwen 3 MoE models (#6673)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: repair_correctness; artifacts=L1.ep.deepep_dispatcher; DeepEP dispatcher fixes Qwen3 MoE error.
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+9/-6)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-c087ddd686  (L1, 2025-05-28, sha c087ddd6865a, PR #6627)
TITLE: Refine pre_reorder_triton_kernel slightly to improve performance (#6627)
SOURCES: path_core
STAGE1: optimize; artifacts=L1.ep.layer; EP pre/post reorder Triton kernels refined for performance.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+9/-4); benchmark/kernels/fused_moe_triton/benchmark_ep_pre_reorder_triton.py (+100/-0)
LABELS: high priority
PERF_LINES: Per benchmark result the kernel gains 10-15% performance.
BODY: ## Motivation ⏎  ⏎ In ep_moe kernel _pre_reorder_triton_kernel_ and _post_reorder_triton_kernel_, every inner loop recomputes ⏎ offset = start_offset + tl.arange(...) ⏎  ⏎ The optimization is to create a constant once: ⏎ vec = tl.arange(0, BLOCK_SIZE) ⏎ and inside the loop use idx = start_offset + vec ⏎  ⏎ The benefit is one less instruction each iteration. The warp scheduler can vectorize the access pattern. ⏎ Per benchmark result the kernel gains 10-15% performance. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b581b22504  (L1, 2025-05-30, sha b581b2250474, PR #6772)
TITLE: Fix one bug in the grouped-gemm triton kernel (#6772)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.ep.layer; Grouped-GEMM Triton EP kernel bug fix for unquantized path.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ When block quant is not in use, EPMoE returns incorrect responses. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-ced3c07afe  (L1, 2025-05-30, sha ced3c07afe02, PR #6782)
TITLE: Support token-level quantization for EP MoE (#6782)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: extend_support; artifacts=L1.ep.layer; EP MoE kernels/layer add token-level quantization support.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+18/-2); python/sglang/srt/layers/moe/ep_moe/layer.py (+71/-23)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-55444ed667  (L1, 2025-06-01, sha 55444ed66715, PR #6699)
TITLE: [EP] Add cuda kernel for moe_ep_pre_reorder (#6699)
SOURCES: path_core
STAGE1: introduce; artifacts=L1.ep.reorder_aot; Introduces CUDA moe_ep_pre_reorder kernel replacing Triton path.
ARTIFACT_HINTS: L1.ep.reorder_aot
FILES: sgl-kernel/csrc/moe/ep_moe_reorder_kernel.cu (+89/-0); sgl-kernel/python/sgl_kernel/moe.py (+24/-0); sgl-kernel/CMakeLists.txt (+1/-0); sgl-kernel/benchmark/bench_moe_ep_pre_reorder.py (+100/-0); sgl-kernel/csrc/common_extension.cc (+4/-0); sgl-kernel/include/sgl_kernel_ops.h (+11/-0); sgl-kernel/python/sgl_kernel/__init__.py (+1/-0)
PERF_LINES: The new kernel gains 10-20% performance improvement.
BODY: ## Motivation ⏎  ⏎ moe_pre_reorder is one of the important kernels in EP MoE. ⏎ Currently moe_pre_reorder is using triton kernel. This PR is to introduce cuda implementation for this kernel. ⏎ The new kernel gains 10-20% performance improvement. ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-eb38c7d1ca  (L1, 2025-06-02, sha eb38c7d1cae1, PR #6093)
TITLE: [1/2] Add Kernel support for Cutlass based Fused FP4 MoE (#6093)
SOURCES: path_core
STAGE1: introduce; artifacts=L1.cutlass.nvfp4,L1.cutlass.adapters; Adds CUTLASS NVFP4/FP4 fused MoE kernel and adapter hooks.
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise, L1.cutlass.nvfp4, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+178/-1); sgl-kernel/csrc/moe/nvfp4_blockwise_moe.cu (+471/-0); sgl-kernel/csrc/moe/prepare_moe_input.cu (+145/-19); sgl-kernel/python/sgl_kernel/moe.py (+55/-0); python/sglang/test/test_fp4_moe.py (+247/-0); sgl-kernel/CMakeLists.txt (+2/-0); sgl-kernel/csrc/common_extension.cc (+21/-2); sgl-kernel/csrc/gemm/nvfp4_expert_quant.cu (+431/-0); sgl-kernel/csrc/gemm/nvfp4_quant_entry.cu (+23/-0); sgl-kernel/include/sgl_kernel_ops.h (+24/-0); sgl-kernel/python/sgl_kernel/__init__.py (+3/-0); sgl-kernel/python/sgl_kernel/gemm.py (+77/-0)
LABELS: high priority
PERF_LINES: Times are in milliseconds (ms).
BODY: This kernel adds support for NVFP4 MoE kernels.  ⏎  ⏎ Currently measured perf:  ⏎ ``` ⏎ [--------------------------------------------------------------------------------------------- FP4 MOE vs FP8 Triton ---------------------------------------------------------------------------------------------] ⏎                                                                                                                        |  triton_moe  |  triton_moe_cuda_graphs  |  cutlass_moe_fp4  |  cutlass_moe_fp4_cuda_graphs ⏎ 1 threads: -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ⏎       nvidia/DeepSeek-R1-FP4, num_experts=256, topk=8, per_act_token=False per_out_ch=False, MKN=((4, 2048, 7168))     |     10.7     |            9.9           |        12.3       |              10.5 ⏎       nvidia/DeepSeek-R1-FP4, num_experts=256, topk=8, per_act_token=False per_out_ch=False, MKN=((8, 2048, 7168))     |     17.5     |           16.7           |        16.4       |              14.5 ⏎       nvidia/DeepSeek-R1-FP4, num_experts=256, topk=8, per_act_token=False per_out_ch=False, MKN=((16, 2048, 7168))    |     29.5     |           28.8           |        23.4       |              21.5 ⏎       nvidia/DeepSeek-R1-FP4, num_experts=256, topk=8, per_act_token=False per_out_ch=False, MKN=((32, 2048, 7168))    |     45.8     |           45.1           |        32.8       |              31.2 ⏎       nvidia/DeepSeek-R1-FP4, num_experts=256, topk=8, per_act_token=False per_out_ch=False, MKN=((64, 2048, 7168))    |     60.9     |           60.0           |        41.5       |              40.3 ⏎       nvidia/DeepSeek-R1-FP4, num_experts=256, topk=8, per_act_token=False per_out_ch=False, MKN=((128, 2048, 7168))   |     69.5     |           68.8           |        47.8       |              45.3 ⏎       nvidia/DeepSeek-R1-FP4, num_experts=256, topk=8, per_act_token=False per_out_ch=False, MKN=((256, 2048, 7168))   |     72.3     |           71.4           |        50.9       |              49.9 ⏎       nvidia/DeepSeek-R1-FP4, num_experts=256, topk=8, per_act_token=False per_out_ch=False, MKN=((512, 2048, 7168))   |     80.9     |           80.1           |        59.6       |              57.8 ⏎       nvidia/DeepSeek-R1-FP4, num_experts=256, topk=8, per_act_token=False per_out_ch=False, MKN=((1024, 2048, 7168))  |     93.7     |           92.8           |        75.9       |              74.7 ⏎       nvidia/DeepSeek-R1-FP4, num_experts=256, topk=8, per_ …[truncated]

### L1-ff00895c46  (L1, 2025-06-02, sha ff00895c46a4, PR #6456)
TITLE: Add CPU optimized kernels for topk and rope fusions  (#6456)
SOURCES: path_core, symbol_pickaxe
STAGE1: introduce; artifacts=L1.hardware.cpu_npu_musa; Adds CPU TopK+sigmoid and softmax+TopK optimized kernels.
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: sgl-kernel/csrc/cpu/topk.cpp (+221/-0); sgl-kernel/csrc/cpu/norm.cpp (+77/-0); sgl-kernel/csrc/cpu/rope.cpp (+310/-93); sgl-kernel/csrc/cpu/torch_extension_cpu.cpp (+25/-4); test/srt/cpu/test_norm.py (+14/-0); test/srt/cpu/test_rope.py (+103/-5); test/srt/cpu/test_topk.py (+83/-0)
LABELS: sgl-kernel, intel, cpu
BODY: ## Overview: ⏎  ⏎ This PR is adding the following CPU optimized sgl-kernels that are at least used in Qwen3/LLama1-4 models: ⏎ ``` ⏎ - TopK fusions:  ⏎       TopK+sigmoid ⏎       softmax+TopK ⏎ - Norm fusion: ⏎       L2norm ⏎ - RoPE fusions: ⏎       origin rope fusion (gpt_neox style and gptj style) ⏎ ```

### L1-8a5480528d  (L1, 2025-06-03, sha 8a5480528d71, PR #6735)
TITLE: [Refactor] Rename `n_share_experts_fusion` as `num_fused_shared_experts` (#6735)
SOURCES: path_core, symbol_pickaxe
STAGE1: adapt_framework; artifacts=L1.routing.topk_py,L1.routing.fused_gate; Renames shared-expert fusion parameter through topk and fused_gate API.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.routing.topk_py, L1.routing.fused_gate
FILES: python/sglang/srt/layers/moe/topk.py (+15/-15); sgl-kernel/csrc/moe/moe_fused_gate.cu (+13/-13); sgl-kernel/python/sgl_kernel/moe.py (+3/-3); benchmark/kernels/fused_moe_triton/README.md (+4/-7); benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py (+3/-7); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+2/-10); python/sglang/srt/managers/schedule_batch.py (+1/-1); python/sglang/srt/model_executor/model_runner.py (+1/-1); python/sglang/srt/models/deepseek_nextn.py (+1/-1); python/sglang/srt/models/deepseek_v2.py (+24/-20); python/sglang/srt/server_args.py (+3/-3); sgl-kernel/csrc/common_extension.cc (+1/-1); sgl-kernel/include/sgl_kernel_ops.h (+1/-1); sgl-kernel/tests/test_moe_fused_gate.py (+10/-10)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-81964328b7  (L1, 2025-06-04, sha 81964328b7ed, PR #6736)
TITLE: Set `num_fused_shared_experts` as `num_shared_experts` when shared_experts fusion is not disabled (#6736)
SOURCES: path_core, symbol_pickaxe
STAGE1: extend_support; artifacts=L1.routing.fused_gate,L1.triton.fused_moe; Updates shared-expert fusion count handling in fused_gate/fused_moe paths.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.routing.topk_py, L1.routing.fused_gate, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+3/-0); python/sglang/srt/layers/moe/fused_moe_native.py (+4/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=257,N=128,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+2/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+9/-0); python/sglang/srt/layers/moe/topk.py (+1/-1); sgl-kernel/csrc/moe/moe_fused_gate.cu (+12/-4); sgl-kernel/python/sgl_kernel/moe.py (+2/-2); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+6/-3); python/sglang/srt/layers/quantization/__init__.py (+1/-0); python/sglang/srt/layers/quantization/blockwise_int8.py (+2/-0); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+4/-0); python/sglang/srt/layers/quantization/fp8.py (+2/-0); python/sglang/srt/layers/quantization/moe_wna16.py (+2/-0); python/sglang/srt/layers/quantization/w8a8_fp8.py (+2/-0); python/sglang/srt/layers/quantization/w8a8_int8.py (+2/-0); python/sglang/srt/managers/schedule_batch.py (+1/-1); python/sglang/srt/model_executor/model_runner.py (+1/-1); python/sglang/srt/models/deepseek_v2.py (+27/-24); python/sglang/srt/server_args.py (+4/-7); sgl-kernel/tests/test_moe_fused_gate.py (+2/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎  ⏎ ## TODO ⏎  ⏎ Finetune kernel configs ⏎  ⏎ ## Checklist

### L1-bd75690f4e  (L1, 2025-06-04, sha bd75690f4eef, PR #6858)
TITLE: fix ep_moe_reorder kernel bugs (#6858)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L1.ep.reorder_aot; Fixes bugs in CUDA ep_moe_reorder kernel.
ARTIFACT_HINTS: L1.ep.reorder_aot
FILES: sgl-kernel/csrc/moe/ep_moe_reorder_kernel.cu (+35/-24); sgl-kernel/benchmark/bench_moe_ep_pre_reorder.py (+10/-7); sgl-kernel/tests/test_ep_moe_pre_reorder_kernel.py (+181/-0)
BODY: ## Motivation ⏎  ⏎  ⏎ ## benchmark in h100 ⏎  ⏎ ```shell ⏎ ep-moe-pre-reorder-performance: ⏎    batch_size  CUDA Kernel  Triton Kernel ⏎ 0        64.0     9.952000      15.584000 ⏎ 1       128.0    10.144000      15.712000 ⏎ 2       256.0    12.864000      17.440001 ⏎ 3       512.0    17.824000      22.528000 ⏎ 4       640.0    23.712000      24.831999 ⏎ 5       768.0    24.896000      27.456000 ⏎ 6      1024.0    29.247999      33.920001 ⏎ 7      2048.0    55.039998      66.399999 ⏎ 8      4096.0   108.287998     147.551998 ⏎ ``` ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-499f5e620c  (L1, 2025-06-04, sha 499f5e620c24, PR #6878)
TITLE: Fix one missing arg in DeepEP (#6878)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: repair_correctness; artifacts=L1.ep.layer; EP MoE layer fixes missing DeepEP argument.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+22/-19)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-5aff1e9392  (L1, 2025-06-05, sha 5aff1e9392d0, PR #6820)
TITLE: Fix Qwen3MoE missing token padding optimization (#6820)
SOURCES: path_core
STAGE1: repair_performance; artifacts=L1.routing.topk_py; Qwen3 topk.py restores token-padding optimization.
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+3/-3); python/sglang/srt/models/qwen3_moe.py (+2/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-43baba649e  (L1, 2025-06-05, sha 43baba649e43, PR #6837)
TITLE: [EP] Add cuda kernel for moe_ep_post_reorder (#6837)
SOURCES: path_core
STAGE1: introduce; artifacts=L1.ep.reorder_aot; Adds CUDA moe_ep_post_reorder kernel path.
ARTIFACT_HINTS: L1.ep.reorder_aot
FILES: sgl-kernel/csrc/moe/ep_moe_reorder_kernel.cu (+83/-2); sgl-kernel/python/sgl_kernel/moe.py (+22/-0); sgl-kernel/benchmark/bench_moe_ep_post_reorder.py (+92/-0); sgl-kernel/csrc/common_extension.cc (+6/-2); sgl-kernel/include/sgl_kernel_ops.h (+10/-0); sgl-kernel/python/sgl_kernel/__init__.py (+1/-0); sgl-kernel/tests/test_ep_moe_post_reorder_kernel.py (+163/-0)
LABELS: ready-to-merge
BODY: ## Motivation ⏎ moe_post_reorder is one of the important kernels in EP MoE. ⏎ Currently moe_post_reorder is using triton kernel. This PR is to introduce CUDA implementation for this kernel. ⏎ The new kernel is expected to gain performance improvement. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-0df6765c83  (L1, 2025-06-05, sha 0df6765c83e2, PR #6887)
TITLE: [CUTLASS-FP4-MOE]  Introduce CutlassMoEParams class for easy initialization of Cutlass Grouped Gems Metadata (#6887)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: adapt_framework; artifacts=L1.cutlass.adapters; CutlassMoEParams changes CUTLASS MoE adapter metadata initialization.
ARTIFACT_HINTS: L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+39/-54); python/sglang/srt/layers/moe/cutlass_moe_params.py (+169/-0); python/sglang/srt/layers/quantization/fp8.py (+2/-2); sgl-kernel/python/sgl_kernel/moe.py (+7/-11); python/sglang/test/test_cutlass_moe.py (+3/-3); python/sglang/test/test_fp4_moe.py (+10/-9)
PERF_LINES: - [N/A] Provide throughput / latency benchmark results and accuracy evaluation results as needed, according to [Benchmark and Profiling](https://docs.sglang.ai/
BODY: ## Motivation ⏎  ⏎ Refactors Cutlass MoE to keep the interface cleaner. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ Introduces `CutlassMoEParams` Class that creates all the cutlass metadata based on the shape of intermediate size and hidden shape ⏎  ⏎ ## Checklist ⏎  ⏎ - [N/A] Update documentation / docstrings / example tutorials as needed, according to [Writing Documentation](https://docs.sglang.ai/references/contribution_guide.html#writing-documentation-running-docs-ci). ⏎ - [N/A] Provide throughput / latency benchmark results and accuracy evaluation results as needed, according to [Benchmark and Profiling](https://docs.sglang.ai/references/benchmark_and_profiling.html) and [Accuracy Results](https://docs.sglang.ai/references/accuracy_evaluation.html).
