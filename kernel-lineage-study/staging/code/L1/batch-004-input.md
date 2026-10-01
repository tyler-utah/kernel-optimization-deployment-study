### L1-253454de9b  (L1, 2025-07-06, sha 253454de9b53, PR #7689)
TITLE: Integrate triton moe kernel (#7689)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: integrate; artifacts=L1.runner.openai_triton_kernels,L1.upstream.openai_triton_kernels; Adds triton_kernels_moe adapter and server option for OpenAI Triton MoE kernel.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.runner.openai_triton_kernels, L1.upstream.openai_triton_kernels
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+2/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+95/-54); python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py (+176/-0); python/sglang/srt/server_args.py (+6/-0); benchmark/kernels/fused_moe_triton/benchmark_sglang_fused_moe_triton.py (+271/-0); python/sglang/srt/managers/schedule_batch.py (+1/-0); test/srt/test_triton_fused_moe.py (+146/-0)
PERF_LINES: Running MoE tests: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
BODY: ## Motivation ⏎  ⏎ This PR is to follow up https://github.com/sgl-project/sglang/issues/7287. ⏎ The main purpose is to replace fused_moe with Triton v3.4.0 matmul_ogs kernel, in order to improve fused_moe's performance. ⏎  ⏎ ``` ⏎ #python ./benchmark/kernels/fused_moe_triton/benchmark_sglang_fused_moe_triton.py --use-cuda-graph ⏎ INFO 07-01 07:27:05 [__init__.py:244] Automatically detected platform cuda. ⏎ shape_configs={'num_experts': 128, 'topk': 8, 'hidden_size': 2048, 'shard_intermediate_size': 768, 'dtype': torch.bfloat16, 'block_shape': None} ⏎ benchmark sglang_fused_moe_triton_v340 with batch_size=128 ⏎ benchmark sglang_fused_moe_triton with batch_size=128 ⏎ Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /usr/local/lib/python3.10/dist-packages/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=384,device_name=NVIDIA_L20Y.json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton ⏎ benchmark sglang_fused_moe_triton_v340 with batch_size=256 ⏎ benchmark sglang_fused_moe_triton with batch_size=256 ⏎ benchmark sglang_fused_moe_triton_v340 with batch_size=512 ⏎ benchmark sglang_fused_moe_triton with batch_size=512 ⏎ benchmark sglang_fused_moe_triton_v340 with batch_size=1024 ⏎ benchmark sglang_fused_moe_triton with batch_size=1024 ⏎ benchmark sglang_fused_moe_triton_v340 with batch_size=2048 ⏎ benchmark sglang_fused_moe_triton with batch_size=2048 ⏎ benchmark sglang_fused_moe_triton_v340 with batch_size=4096 ⏎ benchmark sglang_fused_moe_triton with batch_size=4096 ⏎ benchmark sglang_fused_moe_triton_v340 with batch_size=8192 ⏎ benchmark sglang_fused_moe_triton with batch_size=8192 ⏎ fused-moe-performance: ⏎    batch_size  sglang_fused_moe_triton_v340  sglang_fused_moe_triton ⏎ 0       128.0                      0.259136                 0.241552 ⏎ 1       256.0                      0.255424                 0.303200 ⏎ 2       512.0                      0.269712                 0.342176 ⏎ 3      1024.0                      0.303376                 0.395168 ⏎ 4      2048.0                      0.366592                 0.516064 ⏎ 5      4096.0                      0.530272                 0.881728 ⏎ 6      8192.0                      0.891840                 1.615424 ⏎ ``` ⏎ ![image](https://github.com/user-attachments/assets/fc414db6-7efe-4d93-9e38-72b4011b1b88) ⏎ Unit test precision check passed: ⏎ ``` ⏎ [root@b75ba454e0f7 /sgl-workspace/sglang] ⏎ #python ./test/srt/test_triton_fused_moe.py ⏎ INFO 07-02 09:14:45 [__init__.py:244] Automatically detected platform cuda. ⏎ Running MoE …[truncated]

### L1-a3398d8478  (L1, 2025-07-07, sha a3398d8478e9, PR #7794)
TITLE: Optimize moe align block size kernel (#7794)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: optimize; artifacts=L1.align.cuda_aot; Optimizes CUDA moe_align with Blelloch scan and reduced global memory access.
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/csrc/moe/moe_align_kernel.cu (+94/-63); sgl-kernel/include/utils.h (+6/-0)
BODY: ## Motivation ⏎  ⏎ - Optimize prefix sum with Blelloch scan (O(logN)) ⏎ - Reduce global memory access ⏎  ⏎ main: ⏎ ``` ⏎     num_tokens  num_experts  topk        SGL  SGL Fusion      Triton ⏎ 0          1.0        128.0   1.0  22.688000   20.624001   32.384001 ⏎ 1          1.0        128.0   2.0  22.688000   20.800000   32.512002 ⏎ 2          1.0        128.0   4.0  22.688000   20.768000   32.543998 ⏎ 3          1.0        128.0   8.0  22.528000   20.640001   32.416001 ⏎ 4          1.0        256.0   1.0  25.823999   23.808001   46.208002 ⏎ 5          1.0        256.0   2.0  25.855999   23.936000   46.239998 ⏎ 6          1.0        256.0   4.0  25.888000   23.903999   46.144001 ⏎ 7          1.0        256.0   8.0  25.760001   23.903999   46.239998 ⏎ 8          8.0        128.0   1.0  22.528000   20.640001   32.543998 ⏎ 9          8.0        128.0   2.0  22.560000   20.640001   32.639999 ⏎ 10         8.0        128.0   4.0  22.560000   20.608000   32.671999 ⏎ 11         8.0        128.0   8.0  22.624001   20.959999   32.960001 ⏎ 12         8.0        256.0   1.0  25.855999   24.192000   46.303999 ⏎ 13         8.0        256.0   2.0  25.792001   24.127999   46.271998 ⏎ 14         8.0        256.0   4.0  25.599999   24.064001   46.399999 ⏎ 15         8.0        256.0   8.0  25.823999   24.160000   46.528000 ⏎ 16        16.0        128.0   1.0  22.560000   20.703999   32.607999 ⏎ 17        16.0        128.0   2.0  22.560000   20.640001   32.639999 ⏎ 18        16.0        128.0   4.0  22.624001   20.864001   32.864001 ⏎ 19        16.0        128.0   8.0  22.560000   21.120001   32.896001 ⏎ 20        16.0        256.0   1.0  25.760001   24.032000   46.367999 ⏎ 21        16.0        256.0   2.0  25.760001   24.127999   46.496000 ⏎ 22        16.0        256.0   4.0  26.016001   24.288001   46.560001 ⏎ 23        16.0        256.0   8.0  26.016001   24.704000   46.688002 ⏎ 24        32.0        128.0   1.0  22.560000   20.640001   32.575998 ⏎ 25        32.0        128.0   2.0  22.655999   20.864001   32.864001 ⏎ 26        32.0        128.0   4.0  22.592001   21.056000   32.928001 ⏎ 27        32.0        128.0   8.0  22.784000   21.504000   34.208000 ⏎ 28        32.0        256.0   1.0  25.792001   24.127999   46.239998 ⏎ 29        32.0        256.0   2.0  25.664000   24.192000   46.496000 ⏎ 30        32.0        256.0   4.0  25.920000   24.607999   46.432000 ⏎ 31        32.0        256.0   8.0  26.016001   25.376000   45.568001 ⏎ 32        64.0        128.0   1.0  22.624001   20.864001   33.008000 ⏎ 33        64.0        128.0   2.0  22.655999   21.183999   32.896001 ⏎ 34        64.0        128.0   4.0  2 …[truncated]

### L1-cb9d91ea8a  (L1, 2025-07-07, sha cb9d91ea8a71, PR #7762)
TITLE: feat: support DeepSeek-R1-W4AFP8 model with ep-moe mode (#7762)
SOURCES: path_core
STAGE1: integrate; artifacts=L1.cutlass.adapters,L1.ep.layer; Adds cutlass_w4a8_moe adapter and wires it into EP MoE mode.
ARTIFACT_HINTS: L1.ep.layer, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_w4a8_moe.py (+215/-0); python/sglang/srt/layers/moe/ep_moe/kernels.py (+58/-0); python/sglang/srt/layers/moe/ep_moe/layer.py (+140/-2); python/sglang/srt/configs/model_config.py (+12/-1); python/sglang/srt/layers/quantization/__init__.py (+2/-0); python/sglang/srt/layers/quantization/fp8.py (+27/-6); python/sglang/srt/layers/quantization/w4afp8.py (+264/-0); python/sglang/srt/models/deepseek_v2.py (+6/-0); python/sglang/srt/server_args.py (+1/-0); python/sglang/test/test_cutlass_w4a8_moe.py (+281/-0)
LABELS: high priority
PERF_LINES: Due to the reduced space required for model weights and decreased bandwidth usage, DeepSeek R1 models can now be run on a single H2O or H100, leading to improve | Request throughput (req/s):              1.12 | Input token throughput (tok/s):          1115.34 | Output token throughput (tok/s):         1115.34 | Total token throughput (tok/s):          2230.68 | ----------------End-to-End Latency--
BODY: ## Motivation ⏎  ⏎  ⏎ This PR supports running [DeepSeek-R1-W4AFP8](https://huggingface.co/Barrrrry/DeepSeek-R1-W4AFP8) model with ep-moe mode(deepep mode support is on the way~) ⏎ Due to the reduced space required for model weights and decreased bandwidth usage, DeepSeek R1 models can now be run on a single H2O or H100, leading to improved throughput and latency. ⏎  ⏎ ## Usage: ⏎ Run without mtp: ⏎ ``` ⏎ SGL_ENABLE_JIT_DEEPGEMM=1 python3 -m sglang.launch_server --model-path ${MODEL_PATH} --context-length 8192 --tp 8 --trust-remote-code --host 0.0.0.0 --port 8000 --mem-fraction-static 0.8 --enable-ep-moe --cuda-graph-max-bs 256 --cuda-graph-bs 1 2 4 8 16 32 64 128 256 --max-running-requests 256 --disable-radix-cache ⏎ ``` ⏎ Run with mtp: ⏎ ``` ⏎ SGL_ENABLE_JIT_DEEPGEMM=1 python3 -m sglang.launch_server --model-path ${MODEL_PATH} --context-length 8192 --tp 8 --trust-remote-code --host 0.0.0.0 --port 8000 --mem-fraction-static 0.8 --enable-ep-moe --cuda-graph-max-bs 256 --cuda-graph-bs 1 2 4 8 16 32 64 128 256 --max-running-requests 256 --disable-radix-cache --speculative-algorithm NEXTN --speculative-draft ${DRAFT_MODEL_PATH} --speculative-num-steps 3 --speculative-eagle-topk 2 --speculative-num-draft-tokens 4 ⏎ ``` ⏎ Note: DRAFT_MODEL can be exported using [export_deepseek_nextn.py script](https://github.com/sgl-project/sglang/blob/main/scripts/export_deepseek_nextn.py) ⏎  ⏎ ## Benchmark ⏎ ### Performance: ⏎ We run  DeepSeek-R1-W4AFP8 on 8\*H20 with ep8, comparing to run DeepSeek-R1 on 16\*H20 with ep16. ⏎ Test configuration: input/output len = 1000/1000, qps=64, max_concurrency=64, num_prompt=256. ⏎ The results are shown below: ⏎  ⏎ DeepSeek-R1-W4AFP8 on 8*H20 with ep8 ⏎ - without mtp ⏎ ``` ⏎ ============ Serving Benchmark Result ============ ⏎ Backend:                                 sglang ⏎ Traffic request rate:                    64.0 ⏎ Max request concurrency:                 64 ⏎ Successful requests:                     256 ⏎ Benchmark duration (s):                  229.53 ⏎ Total input tokens:                      256000 ⏎ Total generated tokens:                  256000 ⏎ Total generated tokens (retokenized):    255367 ⏎ Request throughput (req/s):              1.12 ⏎ Input token throughput (tok/s):          1115.34 ⏎ Output token throughput (tok/s):         1115.34 ⏎ Total token throughput (tok/s):          2230.68 ⏎ Concurrency:                             63.86 ⏎ ----------------End-to-End Latency---------------- ⏎ Mean E2E Latency (ms):                   57252.75 ⏎ Median E2E Latency (ms):                 57075.26 ⏎ ---------------Time to First Token---------------- ⏎ Mean TTFT (ms):   …[truncated]

### L1-659907e32b  (L1, 2025-07-08, sha 659907e32b95, PR #7129)
TITLE: Enable ModelOpt Llama4 fp8 checkpoint deployment in SGLang (#7129)
SOURCES: path_core, symbol_pickaxe
STAGE1: extend_support; artifacts=L1.triton.fused_moe; Extends FusedMoE layer for ModelOpt Llama4 FP8 MoE checkpoint deployment.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+39/-1); python/sglang/srt/layers/quantization/modelopt_quant.py (+244/-1); python/sglang/srt/models/mllama4.py (+360/-79)
LABELS: high priority
PERF_LINES: The development of humanoid robots like Sophia represents a significant advancement in artificial intelligence (AI). These robots are designed to resemble human
BODY: ## Motivation ⏎  ⏎ Enable ModelOpt Llama4 fp8 checkpoint deployment in SGLang, as part of our efforts to promote ModelOpt in SGLang. See https://github.com/sgl-project/sglang/issues/5251 ⏎  ⏎ Resolve request in https://github.com/NVIDIA/TensorRT-Model-Optimizer/issues/203  ⏎  ⏎ ## Modifications ⏎  ⏎ - Introduced `ModelOptFp8MoEMethod` to support Llama 4 FP8 MoE: ⏎   - Handles weight and scale creation, post-processing, and kernel invocation.  ⏎ - Enhanced fused MoE layer to support ModelOpt scalers. ⏎ - Enabled loading of ModelOpt Llama 4 FP8 checkpoints: ⏎   - Added support for module name transformation and matching. ⏎   - Implemented FP8 scale loading for BMM-style experts. ⏎   - Refactored `Llama4ForConditionalGeneration.load_weights`: ⏎     - Extracted helper methods to improve modularity and support diverse loading cases.  ⏎   ⏎ - Testing script: ⏎ ``` ⏎ import sglang as sgl ⏎  ⏎ def main(): ⏎     prompts = [ ⏎         "Hello, my name is", ⏎         "The president of the United States is", ⏎         "The capital of France is", ⏎         "The future of AI is", ⏎     ] ⏎  ⏎     sampling_params = { ⏎         "temperature": 0.7, ⏎         "top_p": 0.9, ⏎         "max_new_tokens": 128, ⏎         "skip_special_tokens": True ⏎     } ⏎  ⏎     llm = sgl.Engine( ⏎         model_path="/home/scratch.omniml_data_2/zhiyuc/checkpoints/Llama-4-Scout-17B-16E-Instruct-fp8", ⏎         quantization="modelopt", ⏎         tp_size=8, ⏎         context_length=4096 ⏎     ) ⏎  ⏎     outputs = llm.generate(prompts, sampling_params) ⏎  ⏎     for prompt, output in zip(prompts, outputs): ⏎         print("=" * 50) ⏎         print(f"Prompt: {prompt}") ⏎         print(f"Generated: {output['text']}") ⏎         print() ⏎  ⏎ if __name__ == "__main__": ⏎     main() ⏎ ``` ⏎ Outputs: ⏎ ``` ⏎ ================================================== ⏎ Prompt: Hello, my name is ⏎ Generated: . I'm here to help you with your questions and provide information on a wide range of topics. How can I assist you today? ⏎  ⏎ ================================================== ⏎ Prompt: The president of the United States is ⏎ Generated:  the head of state and head of government of the United States, and the highest-ranking official in the executive branch of the federal government. The president is also the commander-in-chief of the United States Armed Forces and the leader of the free world. ⏎  ⏎ The president is elected through the Electoral College system, where each state is allocated a certain number of electoral votes based on its population. The candidate who wins the most votes in a state gets all of that state's electoral votes, except in Maine and Nebraska whi …[truncated]

### L1-128f16a817  (L1, 2025-07-08, sha 128f16a81728, PR #7818)
TITLE: [CPU]convert topk_weights to fp32 for INT8 and FP8 paths (for llama4) and fix LmHead weight pack (#7818)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.hardware.cpu_npu_musa; Fixes CPU MoE INT8/FP8 topk_weights dtype and CPU moe.cpp path.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-3); sgl-kernel/csrc/cpu/moe.cpp (+10/-5); python/sglang/srt/layers/vocab_parallel_embedding.py (+9/-3)
LABELS: intel, cpu
BODY: ## Motivation ⏎  ⏎  ⏎ 1. Convert `topk_weights` to fp32 for INT8 and FP8 paths since the CPU kernel requires it to be fp32. ⏎ Previously we've fixed this issue in BF16 path for `llama4`. The same fix is needed for INT8 and FP8. ⏎ https://github.com/sgl-project/sglang/blob/3646f6bb3e42ef31b29e4bae3244a14333a2ba9b/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L320-L322 ⏎  ⏎ 2. For the weight pack check in `ParallelLMHead`, we previously checked that `self.quant_config` should be None to ensure it's not quantized. But `self.quant_config` could be `Fp8Config` for FP8 model or `W8A8Int8Config` for INT8 model while the weight of `ParallelLMHead` is still `torch.bfloat16`. We updated the code to explicitly check the weight dtype to device whether weight pack is supported or not on CPU.

### L1-d389bedf72  (L1, 2025-07-09, sha d389bedf72a6, PR #7838)
TITLE: [CPU][Qwen3 MoE] Enable fused_topk CPU fusion and enhance FP8 TP padding (#7838)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: change_default; artifacts=L1.hardware.cpu_npu_musa,L1.routing.topk_py; Enables fused_topk CPU fusion for Qwen3 FP8 CPU MoE path.
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+7/-1); python/sglang/srt/layers/parameter.py (+19/-3)
LABELS: ready-to-merge, intel, cpu
BODY: This PR contains the following fixes when run with fp8 MoE models on CPUs like https://huggingface.co/Qwen/Qwen3-30B-A3B-FP8   ⏎  ⏎ 1. Enable fused_topk CPU fusion ⏎    ->  `fused_topk_cpu ` could cover the original `fused_topk `func on CPU path ⏎ 2. fix `load_qkv_weight `loading when odd TP size ⏎    -> When running with TP size that is not dividable, refering to other weight loader, adding `narrow_padded_param_and_loaded_weight `for `load_qkv_weight ` as well.

### L1-766392c6bd  (L1, 2025-07-10, sha 766392c6bda2, PR #7791)
TITLE: [feature]Ascend quantization support (#7791)
SOURCES: path_core, symbol_pickaxe
STAGE1: extend_support; artifacts=L1.hardware.cpu_npu_musa,L1.routing.topk_py,L1.ep.layer; Adds Ascend/NPU quantization support touching MoE topk and EP/fused layers.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+2/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-1); python/sglang/srt/layers/moe/topk.py (+1/-1); python/sglang/srt/configs/model_config.py (+3/-1); python/sglang/srt/layers/linear.py (+10/-0); python/sglang/srt/layers/quantization/moe_wna16.py (+1/-2); python/sglang/srt/layers/quantization/w8a8_int8.py (+738/-14); python/sglang/srt/mem_cache/memory_pool.py (+4/-2); python/sglang/srt/model_loader/loader.py (+23/-12); python/sglang/srt/models/llama.py (+2/-0); python/sglang/srt/models/mixtral_quant.py (+4/-0); python/sglang/srt/models/qwen2.py (+2/-0); python/sglang/srt/utils.py (+99/-1)
LABELS: high priority
PERF_LINES: python -m unittest test_w8a8_quantization.TestW8A8.test_throughput | Throughput: 10.277945525301533 tokens/s | Latency: 69.105 s | Output throughput: 344.622 token/s
BODY: ## Motivation ⏎  ⏎ support W8A8 quantized models inference on Ascend servers. ⏎  ⏎ ## Modifications ⏎  ⏎ - quant ⏎  ⏎     Extended W8A8 quantization support with NPU specific quantization configurations for static and dynamic cases. ⏎  ⏎     Added NPU specific quantization support interface classes for Linear layer utilized by respective configurations: ⏎     `NPU_W8A8LinearMethod`, `NPU_W8A8DynamicLinearMethod` ⏎  ⏎     Added NPU specific quantized Linear layer implementations to perform layer inference in accordance with chosen quantization mode and available SW/HW capabilities: `NPU_W8A8LinearMethodImpl`, `NPU_W8A8LinearMethodMTImpl`, `NPU_W8A8DynamicLinearMethodImpl` ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Accuracy and performance result ⏎  ⏎ - quant ⏎     ```shell ⏎     # Due to the issue with quantized weight generation and downloading. UT will be added later ⏎     python -m unittest test_w8a8_quantization.TestW8A8.test_throughput ⏎     ``` ⏎      ⏎     ```shell ⏎     Throughput: 10.277945525301533 tokens/s ⏎     ``` ⏎      ⏎     ```shell ⏎     python -m unittest test_w8a8_quantization.TestW8A8.test_gsm8k ⏎     ``` ⏎     ``` ⏎     Accuracy: 0.715 ⏎     Invalid: 0.045 ⏎     Latency: 69.105 s ⏎     Output throughput: 344.622 token/s ⏎     ```

### L1-191d836ff6  (L1, 2025-07-11, sha 191d836ff616, PR #7953)
TITLE: fix: minor fix for modelopt weight load compatibility (#7953)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.triton.fused_moe; Fixes modelopt weight-load compatibility inside FusedMoE layer.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+6/-1)
BODY: ## Motivation ⏎ fix compatibility when loading weight with different pack method. ⏎ related PR: https://github.com/sgl-project/sglang/pull/7129 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ only apply dim check when using modelopt quantization ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-5f6756b038  (L1, 2025-07-12, sha 5f6756b038ff, PR #7814)
TITLE: [BugFix] fix pre_reorder_triton_kernel default int32 issue (#7814)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.ep.layer; Fixes pre_reorder_triton_kernel index type overflow for large grids.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+4/-2)
BODY: …cause overflow with large grid size ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎ [Bugfix] fix pre_reorder_triton_kernel default int32 issue which may cause overflow with large grid size ⏎  ⏎ ## Modifications ⏎ Modified the default index type for src and dst in pre_reorder_triton_kernel from int32 to int64 to avoid overflow caused by excessively large grid sizes. ⏎ ## Checklist

### L1-d9eb5efc71  (L1, 2025-07-16, sha d9eb5efc71b1, PR #8098)
TITLE: [misc] update nvshmem and pin deepEP commit hash (#8098)
SOURCES: subject_keyword, dependency_pin, release_notes
STAGE1: repair_build_dependency; artifacts=L1.upstream.deepep; Pins DeepEP commit in Docker, changing delivered DeepEP dependency for EP dispatch.
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+6/-6)
LABELS: dependencies
BODY: ## Motivation ⏎  ⏎ Update the Docker build to use the latest NVSHMEM version 3.3.9 and improve reproducibility by pinning the DeepEP dependency to a specific commit hash.  ⏎  ⏎ ## Modifications ⏎  ⏎ - **Updated NVSHMEM version**: Upgraded from 3.2.5 to 3.3.9 ⏎   - Changed download URL to use `nvshmem_src_cuda12-all-all-3.3.9.tar.gz` ⏎   - Updated cleanup commands to match new filename ⏎ - **Pinned DeepEP commit**: Added explicit commit hash `b6ce310bb0b75079682d09bc2ebc063a074fbd58` to ensure reproducible builds ⏎ - **Made DeepEP commit configurable**: Added `DEEPEP_COMMIT` build argument with the pinned commit as default ⏎   - Allows override during build time with `--build-arg DEEPEP_COMMIT=<hash>` ⏎ - **Removed NVSHMEM patches**:  ⏎   - Removed `git apply /sgl-workspace/DeepEP/third-party/nvshmem.patch` ⏎   - Removed `sed` command that added `#include <unistd.h>` to `examples/moe_shuffle.cu` ⏎   - These modifications are no longer needed with NVSHMEM 3.3.9 ⏎  ⏎ ## Checklist

### L1-c28ad1990d  (L1, 2025-07-16, sha c28ad1990d29, PR #7992)
TITLE: [1/n] chore: decouple quantization implementation from vLLM dependency (#7992)
SOURCES: path_core, symbol_pickaxe
STAGE1: repair_build_dependency; artifacts=L1.runner.marlin; Decouples GPTQ/Marlin quantization implementation from vLLM dependency used by MoE quant path.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.marlin
FILES: python/sglang/srt/layers/moe/fused_moe_triton/__init__.py (+4/-1); sgl-kernel/python/sgl_kernel/fused_moe.py (+2/-1); python/sglang/srt/layers/quantization/__init__.py (+2/-4); python/sglang/srt/layers/quantization/gptq.py (+491/-119); python/sglang/srt/layers/quantization/marlin_utils.py (+781/-0); python/sglang/srt/layers/quantization/moe_wna16.py (+30/-0); python/sglang/srt/layers/quantization/quant_utils.py (+0/-166); python/sglang/srt/layers/quantization/scalar_type.py (+0/-0); python/sglang/srt/layers/quantization/utils.py (+162/-1); sgl-kernel/tests/test_marlin_repack.py (+2/-4); test/srt/test_gptqmodel_dynamic.py (+4/-5); test/srt/test_int4_kernel.py (+0/-301); test/srt/test_w4a8.py (+0/-14)
LABELS: high priority
BODY: ## Motivation ⏎ The primary goal of this change is to enhance the consistency and stability of SGLang's quantization features. By decoupling the quantization implementation from its vLLM dependency, we aim to make the module easier to maintain and more portable. ⏎ Full realization of this goal will involve several subsequent PRs; this particular PR addresses the GPTQ feature. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-5c08a36cbf  (L1, 2025-07-16, sha 5c08a36cbfae, PR #8110)
TITLE: [Fix] ensure DeepGEMM is only enabled for FP8_W8A8 models (#8110)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.ep.layer,L1.runner.deep_gemm; Adds EP MoE guard so DeepGEMM only runs for FP8_W8A8 models.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+6/-0)
BODY: ## Motivation ⏎  ⏎ If a user erroneously enables the ENABLE_JIT_DEEPGEMM environment variable in non-FP8 model scenarios (e.g., when using Qwen3-235B-FP16), sglang will enter a failed state during launch and sglang‘s rank scheduler will stuck at the following stack: ⏎  ⏎  ⏎ Thread 66311 (active+gil): "Thread-3 (forward_thread_func)" ⏎     dispatch (deep_ep/buffer.py:349) ⏎     _dispatch_core (ep_moe/token_dispatcher.py:349) ⏎     dispatch_b (ep_moe/token_dispatcher.py:265) ⏎     dispatch_b (ep_moe/token_dispatcher.py:703) ⏎     dispatch (ep_moe/token_dispatcher.py:681) ⏎     _execute (two_batch_overlap.py:795) ⏎     dispatch (two_batch_overlap.py:798) ⏎     forward_deepep (qwen3_moe.py:228) ⏎     forward (qwen3_moe.py:168) ⏎  ⏎  ⏎ **This PR introduces an initialization check to detect such misconfigurations**: when ENABLE_JIT_DEEPGEMM is enabled without an FP8 quantized model present, the system will throw a fatal error with instructions to adjust configuration parameters. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-af1cc8fe2d  (L1, 2025-07-17, sha af1cc8fe2dd8, PR #7884)
TITLE: [kernel] opt moe align block kernel by block/warp scan algorithm (#7884)
SOURCES: path_core, subject_keyword, release_notes, corpus:confirmed-reverts(reverted)
STAGE1: optimize; artifacts=L1.align.cuda_aot; Applies block/warp scan optimization to CUDA moe_align kernel.
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/csrc/moe/moe_align_kernel.cu (+51/-42)
DEEP_STUDY: deep-study: this PR was reverted by PR 8457 (confirmed_revert, reason=hardware_specific_breakage)
PERF_LINES: This PR is to introduce block / warp scan algorithm in fused MoE path **moe_align_block_size_kernel**  which gains approximately 10% speedup. | sgl-kernel/tests/test_moe_align.py ............................................................................................................................. | .............................................................................................
BODY: ## Motivation ⏎  ⏎  ⏎ This PR is to introduce block / warp scan algorithm in fused MoE path **moe_align_block_size_kernel**  which gains approximately 10% speedup. ⏎  ⏎ Here is the benchmark result. ⏎ Note: num_experts >= 128 appliable to this PR. num_experts < 128 appliable to **moe_align_block_size_small_batch_expert_kernel**  kernel. ⏎  ⏎ This PR: ⏎ ``` ⏎ $python ./sgl-kernel/benchmark/bench_moe_align_block_size.py ⏎ INFO 07-09 13:15:28 [__init__.py:244] Automatically detected platform cuda. ⏎ ✅ VLLM implementation works with 64 experts! ⏎ ✅ SGL and Triton implementations match ⏎ ✅ SGL and VLLM implementations match ⏎  ⏎ 📊 Running performance benchmark for 64 experts... ⏎ moe-align-block-size-performance: ⏎      num_tokens  num_experts  topk        SGL  SGL Fusion       Triton ⏎ 0           1.0          8.0   1.0  16.672000   14.912000    44.032000 ⏎ 1           1.0          8.0   2.0  16.896000   15.072000    44.256002 ⏎ 2           1.0          8.0   4.0  16.672000   15.008000    43.264002 ⏎ 3           1.0          8.0   8.0  16.432000   15.104000    44.767998 ⏎ 4           1.0         32.0   1.0  19.200001   17.424000    44.512000 ⏎ 5           1.0         32.0   2.0  19.231999   17.503999    44.464000 ⏎ 6           1.0         32.0   4.0  19.200001   17.535999    43.728001 ⏎ 7           1.0         32.0   8.0  19.231999   17.568000    44.767998 ⏎ 8           1.0         64.0   1.0  22.655999   20.864001    44.160001 ⏎ 9           1.0         64.0   2.0  22.608001   20.927999    44.799998 ⏎ 10          1.0         64.0   4.0  22.624001   20.927999    44.863999 ⏎ 11          1.0         64.0   8.0  22.496000   20.959999    45.343999 ⏎ 12          1.0        128.0   1.0  19.296000   17.696001    47.295999 ⏎ 13          1.0        128.0   2.0  19.264000   17.472001    44.927999 ⏎ 14          1.0        128.0   4.0  19.328000   17.728001    43.968000 ⏎ 15          1.0        128.0   8.0  19.296000   17.503999    44.560000 ⏎ 16          1.0        256.0   1.0  19.231999   17.632000    45.311999 ⏎ 17          1.0        256.0   2.0  19.200001   17.728001    45.056000 ⏎ 18          1.0        256.0   4.0  19.360000   17.440001    46.271998 ⏎ 19          1.0        256.0   8.0  19.328000   17.759999    46.016000 ⏎ 20          8.0          8.0   1.0  16.543999   15.104000    44.096000 ⏎ 21          8.0          8.0   2.0  16.576000   14.976000    44.383999 ⏎ 22          8.0          8.0   4.0  16.319999   15.008000    45.407999 ⏎ 23          8.0          8.0   8.0  17.055999   15.488000    44.911999 ⏎ 24          8.0         32.0   1.0  19.216000   17.568000    45.152001 ⏎ 25          8.0         32 …[truncated]

### L1-48c1fa7bb6  (L1, 2025-07-17, sha 48c1fa7bb695, PR #7889)
TITLE: [CPU][Llama4] Fix Llama4 MoE inputs with "apply_router_weight_on_input"  (#7889)
SOURCES: path_core, symbol_pickaxe
STAGE1: extend_support; artifacts=L1.hardware.cpu_npu_musa,L1.routing.topk_py; Adds CPU Llama4 apply_router_weight_on_input handling in topk path.
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+13/-0); python/sglang/srt/configs/update_config.py (+3/-1); python/sglang/srt/layers/quantization/fp8.py (+6/-0); python/sglang/srt/layers/quantization/unquant.py (+8/-3); python/sglang/srt/layers/quantization/w8a8_int8.py (+5/-0)
LABELS: high priority
DEEP_STUDY: deep-study correctness case sglang:48c1fa7bb6: class=numerical_precision; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: This PR fixes the support of "apply_router_weight_on_input" for llama4 model for the CPU path. ⏎ Differing from other MoE models (Qwen3/DeepSeek), llama4 applies the topk weight [before MoE calculation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama4.py#L153). Here we follow the logic to revise the current CPU path, and added TODO to fuse this processing in MoE kernels next.

### L1-15ad6c9086  (L1, 2025-07-19, sha 15ad6c908670, PR #7966)
TITLE: [1/N] MoE Refactor: refactor `select_experts` (#7966)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: adapt_framework; artifacts=L1.routing.topk_py,L1.triton.fused_moe,L1.ep.layer; Extracts select_experts into topk.py and changes FusedMoE/EPMoE call protocol.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.routing.topk_py, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+13/-74); python/sglang/srt/layers/moe/fused_moe_native.py (+7/-47); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+7/-38); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+6/-29); python/sglang/srt/layers/moe/topk.py (+171/-5); python/sglang/srt/layers/quantization/__init__.py (+9/-23); python/sglang/srt/layers/quantization/awq.py (+8/-31); python/sglang/srt/layers/quantization/base_config.py (+14/-7); python/sglang/srt/layers/quantization/blockwise_int8.py (+7/-28); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+21/-71); python/sglang/srt/layers/quantization/fp8.py (+12/-40); python/sglang/srt/layers/quantization/gptq.py (+8/-27); python/sglang/srt/layers/quantization/modelopt_quant.py (+11/-52); python/sglang/srt/layers/quantization/moe_wna16.py (+8/-26); python/sglang/srt/layers/quantization/unquant.py (+55/-152); python/sglang/srt/layers/quantization/w8a8_fp8.py (+9/-28); python/sglang/srt/layers/quantization/w8a8_int8.py (+14/-75); python/sglang/srt/models/deepseek.py (+9/-6); python/sglang/srt/models/deepseek_v2.py (+22/-30); python/sglang/srt/models/grok.py (+9/-3); python/sglang/srt/models/llama4.py (+11/-11); python/sglang/srt/models/mixtral.py (+9/-2); python/sglang/srt/models/qwen2_moe.py (+9/-5); python/sglang/srt/models/qwen3_moe.py (+13/-18); python/sglang/srt/custom_op.py (+5/-2); python/sglang/srt/layers/linear.py (+1/-1); python/sglang/srt/models/granitemoe.py (+8/-2); python/sglang/srt/models/hunyuan.py (+8/-5); python/sglang/srt/models/olmoe.py (+8/-5); python/sglang/srt/models/phimoe.py (+9/-3); python/sglang/test/test_block_fp8.py (+8/-3); python/sglang/test/test_block_fp8_ep.py (+1/-1); python/sglang/test/test_cutlass_w4a8_moe.py (+1/-3); python/sglang/test/test_fp4_moe.py (+1/-3); test/srt/test_block_int8.py (+8/-3); test/srt/test_fused_moe.py (+15/-4); test/srt/test_int8_kernel.py (+7/-3); test/srt/test_triton_moe_channel_fp8_kernel.py (+7/-3); test/srt/test_triton_moe_wna16.py (+8/-3)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This pull request extracts the `select_experts` computation from within `FusedMoE` and `EPMoE`, moving it outside these modules. This refactoring offers three key benefits: ⏎  ⏎ - Enable gate-router fusion.  ⏎ - Simplifying MoE's input: reducing input number from 16 to 7. ⏎ - Unifying API with DeepEPMoE. ⏎  ⏎ This PR temporarily disables `triton_kernel_moe`, which will be added back later. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-a589a07167  (L1, 2025-07-19, sha a589a0716774, PR #7825)
TITLE: fix moe gate dtype, fix tbo, fix fake dispatch (#7825)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.routing.topk_py; Fixes e_score_correction_bias dtype used by top-k routing.
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+1/-1); python/sglang/srt/eplb/expert_location_dispatch.py (+1/-1); python/sglang/srt/models/deepseek_v2.py (+1/-1)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Fix TBO after #7222 import is_extend_in_batch, add this to TBO batch filter ⏎  ⏎ Fix dtype of e_score_correction_bias, which need to be float32 but bfloat16 now. ⏎  ⏎ reference: https://huggingface.co/deepseek-ai/DeepSeek-V3/tree/main?show_file_info=model-00001-of-000163.safetensors ⏎ ![image](https://github.com/user-attachments/assets/f4181632-e424-4fbe-9dba-849f5dff79b6) ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ Other changes: ⏎  ⏎ Change fake dispatch to make EP more balanced, more test is needed. ⏎ After the modification, the prefill performance is more stable when using fake, but the peak value is not as high as before.  ⏎  ⏎ ## Checklist

### L1-465968b2e3  (L1, 2025-07-21, sha 465968b2e328, PR #8197)
TITLE: Fix dtype error in CI (#8197)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.routing.topk_py; Fixes AITER biased_grouped_topk dtype mismatch in topk.py.
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ https://github.com/sgl-project/sglang/actions/runs/16400875083/job/46340127530#step:5:489 ⏎ ``` ⏎   File "/sglang-checkout/python/sglang/srt/layers/moe/topk.py", line 526, in biased_grouped_topk_gpu ⏎     aiter_biased_grouped_topk( ⏎   File "/sgl-workspace/aiter/aiter/jit/core.py", line 631, in wrapper ⏎     return op(*args, **kwargs) ⏎            ^^^^^^^^^^^^^^^^^^^ ⏎ RuntimeError: gating_output.dtype() == correction_bias.dtype() ⏎ ``` ⏎  ⏎ ref: https://github.com/sgl-project/sglang/pull/7825 ⏎  ⏎ Will remove the type conversion once `bf16bf16fp32` gemm supported.

### L1-c9e8613c97  (L1, 2025-07-21, sha c9e8613c9708, PR #8193)
TITLE: Apply fused sorted token ids padding (#8193)
SOURCES: path_core
STAGE1: optimize; artifacts=L1.triton.fused_moe,L1.triton.moe_align; Applies fused sorted-token-id padding in fused_moe path for accuracy/performance.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+7/-2)
PERF_LINES: Latency: 32.717 s | Output throughput: 4104.390 token/s | |    |   max_concurrency |   input_throughput |   output_throughput |   mean_ttft_ms |   median_ttft_ms |   p99_ttft_ms |   mean_tpot_ms |   median_tpot_ms |    | |    |   max_concurrency |   input_throughput |   output_throughput |   mean_ttft_ms |   median_ttft_ms |   p99_ttft_ms |   mean_tpot_ms |   median_tpot_ms |   
BODY: ## Motivation ⏎  ⏎ Apply https://github.com/sgl-project/sglang/pull/7437. ⏎  ⏎ ## Accuracy ⏎ For `DeepSeek-V3-0324`. ⏎ ``` ⏎ Accuracy: 0.936 ⏎ Invalid: 0.000 ⏎ Latency: 32.717 s ⏎ Output throughput: 4104.390 token/s ⏎ ``` ⏎  ⏎ ## Benchmark ⏎ main branch: ⏎ ``` ⏎ +----+-------------------+--------------------+---------------------+----------------+------------------+---------------+----------------+------------------+---------------+-----------------------+ ⏎ |    |   max_concurrency |   input_throughput |   output_throughput |   mean_ttft_ms |   median_ttft_ms |   p99_ttft_ms |   mean_tpot_ms |   median_tpot_ms |   p99_tpot_ms |   per_user_throughput | ⏎ +====+===================+====================+=====================+================+==================+===============+================+==================+===============+=======================+ ⏎ |  0 |             1.000 |             91.811 |              91.811 |        161.828 |          158.319 |       184.206 |         10.738 |           10.738 |        10.741 |                91.811 | ⏎ +----+-------------------+--------------------+---------------------+----------------+------------------+---------------+----------------+------------------+---------------+-----------------------+ ⏎ |  1 |             4.000 |            306.596 |             306.596 |        449.659 |          320.894 |      1541.033 |         12.605 |           12.466 |        13.667 |                76.649 | ⏎ +----+-------------------+--------------------+---------------------+----------------+------------------+---------------+----------------+------------------+---------------+-----------------------+ ⏎ |  2 |            16.000 |            822.728 |             822.728 |        691.112 |          741.801 |       949.421 |         18.768 |           18.773 |        19.277 |                51.421 | ⏎ +----+-------------------+--------------------+---------------------+----------------+------------------+---------------+----------------+------------------+---------------+-----------------------+ ⏎ |  3 |            32.000 |           1210.724 |            1210.724 |       1040.937 |          971.016 |      1529.293 |         25.405 |           25.399 |        26.258 |                37.835 | ⏎ +----+-------------------+--------------------+---------------------+----------------+------------------+---------------+----------------+------------------+---------------+-----------------------+ ⏎ ``` ⏎  ⏎ this PR: ⏎ ``` ⏎ +----+-------------------+--------------------+---------------------+----------------+------------------+---------------+----------------+---- …[truncated]

### L1-e50109f2ed  (L1, 2025-07-21, sha e50109f2edfe, PR #7484)
TITLE: [AMD] Remove vllm's scaled_fp8_quant and moe_sum when SGLANG_USE_AITER=1 (#7484)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: change_default; artifacts=L1.runner.aiter,L1.triton.fused_moe; When SGLANG_USE_AITER=1, replaces vLLM scaled_fp8_quant/moe_sum with AITER counterparts.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-4); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+21/-5); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+3/-2); python/sglang/srt/layers/quantization/fp8.py (+1/-2); python/sglang/srt/layers/quantization/fp8_kernel.py (+115/-46); python/sglang/srt/layers/quantization/unquant.py (+0/-1); python/sglang/srt/layers/quantization/utils.py (+3/-2); python/sglang/test/test_custom_ops.py (+12/-7)
LABELS: high priority, ready-to-merge
PERF_LINES: Latency: 44.225 s | Output throughput: 3138.438 token/s | Latency: 43.901 s | Output throughput: 3145.905 token/s
BODY: ## Motivation ⏎  ⏎ This is part of the efforts to remove vllm's dependency on ROCm.  ⏎ The vLLM's `scaled_fp8_quant` and `moe_sum` will be replaced with the aiter's counterparts when `SGLANG_USE_AITER=1` ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎  ⏎ ### To run the unit tests for `scaled_fp8_quant`: ⏎ ``` ⏎ /sgl-workspace/sglang/python/test# SGLANG_USE_AITER=1 pytest test_custom_ops.py ⏎ /sgl-workspace/sglang/python/test# SGLANG_USE_AITER=0 pytest test_custom_ops.py ⏎ ``` ⏎  ⏎ ### To run the e2e tests for DeepSeek-V3: ⏎ `/sgl-workspace/sglang/test/srt# SGLANG_USE_AITER=1 python3 -m unittest test_full_deepseek_v3.TestDeepseekV3` ⏎  ⏎ **With original vllm's scaled_fp8_quant and moe_sum (BEFORE)** ⏎ ``` ⏎ acc_length=1.00 ⏎ speed=39.02 token/s ⏎ speed=39.02 ⏎  ⏎ Accuracy: 0.953 ⏎ Invalid: 0.000 ⏎ Latency: 44.225 s ⏎ Output throughput: 3138.438 token/s ⏎ ``` ⏎  ⏎ **With original vllm's scaled_fp8_quant and moe_sum (AFTER)** ⏎ ``` ⏎ acc_length=1.00 ⏎ speed=39.12 token/s ⏎ speed=39.12 ⏎  ⏎ Accuracy: 0.950 ⏎ Invalid: 0.000 ⏎ Latency: 43.901 s ⏎ Output throughput: 3145.905 token/s ⏎ ``` ⏎  ⏎ CC: @HaiShaw

### L1-58c468f404  (L1, 2025-07-25, sha 58c468f4045e, PR #8333)
TITLE: Fix FP4 MoE accuracy from missing routed_scaling_factor (#8333)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: repair_correctness; artifacts=NEW:modelopt_fp4_moe_adapter; Fixes FP4 MoE routed_scaling_factor application in ModelOpt quantization adapter.
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/modelopt_quant.py (+8/-4); python/sglang/srt/server_args.py (+0/-4)
ISSUES: #7166 [Bug] Deepseek R1 FP4 model quality drop
BODY: ## Motivation ⏎  ⏎ This PR fixes the accuracy issues with FP4 MoE. ⏎  ⏎ Fixes https://github.com/sgl-project/sglang/issues/7166 ⏎  ⏎ Before: ⏎ ``` ⏎ sglang (pretrained=nvidia/DeepSeek-R1-0528-FP4,trust_remote_code=True,quantization=modelopt_fp4,tp_size=8,max_model_len=32768,add_bos_token=True,enable_flashinfer_moe=True), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: 512 ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8704|±  |0.0093| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.8681|±  |0.0093| ⏎ ``` ⏎  ⏎ After, using FP4 flashinfer cutlass moe: ⏎ ``` ⏎ sglang (pretrained=nvidia/DeepSeek-R1-0528-FP4,trust_remote_code=True,quantization=modelopt_fp4,tp_size=8,max_model_len=32768,add_bos_token=True,enable_flashinfer_moe=True), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: 512 ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9492|±  |0.0060| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.9454|±  |0.0063| ⏎ ``` ⏎  ⏎ After, using FP4 flashinfer cutlass moe and epmoe: ⏎ ``` ⏎ sglang (pretrained=nvidia/DeepSeek-R1-0528-FP4,trust_remote_code=True,quantization=modelopt_fp4,tp_size=8,max_model_len=32768,add_bos_token=True,enable_flashinfer_moe=True,enable_ep_moe=True), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: 512 ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9545|±  |0.0057| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.9500|±  |0.0060| ⏎ ``` ⏎  ⏎ After, using FP4 cutlass moe: ⏎ ``` ⏎ sglang (pretrained=nvidia/DeepSeek-R1-0528-FP4,trust_remote_code=True,quantization=modelopt_fp4,tp_size=8,max_model_len=32768,add_bos_token=True), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: 512 ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9500|±  |0.0060| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.9477|±  |0.0061| ⏎ ``` ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ A previous PR https://github.com/sgl-project/sglang/pull/6970 moved routed expert scaling into `moe_sum_reduce` and r …[truncated]

### L1-9045cc1eb8  (L1, 2025-07-25, sha 9045cc1eb8da, PR #8353)
TITLE: [torch.compile bug] avoid biased_grouped_topk_impl func repeatedly triggering `torch.compile` in forward pass (#8353)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: repair_performance; artifacts=L1.routing.topk_py; Avoids repeated torch.compile triggering for biased_grouped_topk_impl in topk.py.
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+2/-9); docs/references/hardware.rst (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-85486b6f6f  (L1, 2025-07-27, sha 85486b6f6f72, PR #8036)
TITLE: [NVIDIA] Add Flashinfer MoE blockscale fp8 backend (#8036)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: integrate; artifacts=L1.upstream.flashinfer_moe,L1.ep.layer; Adds FlashInfer blockscale FP8 MoE backend to EP/fused MoE path.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+102/-7); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+9/-7); python/sglang/srt/layers/quantization/modelopt_quant.py (+5/-5); python/sglang/srt/models/deepseek_v2.py (+44/-20); python/sglang/srt/models/qwen2_moe.py (+2/-2); python/sglang/srt/models/qwen3_moe.py (+2/-2); python/sglang/srt/server_args.py (+13/-3); python/sglang/srt/managers/schedule_batch.py (+2/-1)
LABELS: high priority
PERF_LINES: Enable flashinfer moe blockscale fp8 backend for low latency scenario. The e2e perf shows up to 3x improvement (see [here](https://github.com/sgl-project/sglang
BODY: Enable flashinfer moe blockscale fp8 backend for low latency scenario. The e2e perf shows up to 3x improvement (see [here](https://github.com/sgl-project/sglang/pull/8036#issuecomment-3104285508)). ⏎  ⏎ cc. @kushanam @pavanimajety

### L1-bf0f448fe5  (L1, 2025-07-27, sha bf0f448fe5b5, PR #8397)
TITLE: [2/N] MoE Refactor: Unify weight loader and quant methods (#8397)
SOURCES: path_core, symbol_pickaxe
STAGE1: adapt_framework; artifacts=L1.triton.fused_moe,L1.ep.layer; Refactors MoE weight loader and quant methods in FusedMoE/EPMoE layers.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+87/-217); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+31/-43); python/sglang/srt/layers/quantization/fp8.py (+25/-247); python/sglang/srt/layers/quantization/unquant.py (+10/-66); python/sglang/srt/layers/quantization/w4afp8.py (+68/-17)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-4d921f2b79  (L1, 2025-07-27, sha 4d921f2b7916, PR #8405)
TITLE: [hotfix] fix merge conflicts in FlashInferEPMoE (#8405)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.ep.layer; Adds missing FlashInferEPMoE attribute after merge conflict fix.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b3eac168e7  (L1, 2025-07-27, sha b3eac168e7de, PR #8258)
TITLE: Support triton kernels v3.4.0 for fused_moe (#8258)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: adapt_framework; artifacts=L1.runner.openai_triton_kernels,L1.routing.topk_py; Updates triton_kernels_moe and topk adapter for Triton kernels v3.4.0 API.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.runner.openai_triton_kernels, L1.upstream.openai_triton_kernels
FILES: python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py (+11/-8); python/sglang/srt/layers/moe/topk.py (+84/-22); python/sglang/srt/layers/quantization/unquant.py (+14/-10)
LABELS: high priority
PERF_LINES: Capturing batches (bs=1 avail_mem=23.02 GB):  98%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████ | Capturing batches (bs=1 avail_mem=23.02 GB): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████ | [2025-07-22 06:18:53 TP0] Decode batch. #running-req: 1, #token: 66, token
BODY: ## Motivation  ⏎  ⏎ This PR is to follow up https://github.com/sgl-project/sglang/pull/7966 [1/N] MoE Refactor: refactor select_experts. It supports triton_kernels v3.4.0 for fused moe. ⏎ This PR involves a change to the TopKOutput data structure.  ⏎  ⏎  ⏎  ⏎ Note: it requires pytorch-triton. ⏎ ``` ⏎ Name: pytorch-triton ⏎ Version: 3.4.0+gitae848267 ⏎ ``` ⏎  ⏎ The result is as expected. ⏎ ``` ⏎ ➜  python git:(support_triton_moe) ✗ python3 -m sglang.launch_server --model Qwen/Qwen3-30B-A3B --tp-size 8 --port 30000 --enable-triton-kernel-moe ⏎ ...... ⏎ [2025-07-22 06:17:55 TP0] Capture cuda graph begin. This can take up to several minutes. avail mem=30.36 GB ⏎ [2025-07-22 06:17:55 TP0] Capture cuda graph bs [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512] ⏎ Capturing batches (bs=1 avail_mem=23.02 GB):  98%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊   | 50/51 [00:36<00:00,  1.45it/s][2025-07-22 06:18:33 TP5] Registering 4947 cuda graph addresses ⏎ Capturing batches (bs=1 avail_mem=23.02 GB): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 51/51 [00:37<00:00,  1.35it/s] ⏎ [2025-07-22 06:18:33 TP1] Registering 4947 cuda graph addresses ⏎ [2025-07-22 06:18:33 TP3] Registering 4947 cuda graph addresses ⏎ [2025-07-22 06:18:33 TP7] Registering 4947 cuda graph addresses ⏎ [2025-07-22 06:18:33 TP6] Registering 4947 cuda graph addresses ⏎ [2025-07-22 06:18:33 TP4] Registering 4947 cuda graph addresses ⏎ [2025-07-22 06:18:33 TP2] Registering 4947 cuda graph addresses ⏎ [2025-07-22 06:18:33 TP0] Registering 4947 cuda graph addresses ⏎ [2025-07-22 06:18:33 TP0] Capture cuda graph end. Time elapsed: 38.61 s. mem usage=7.36 GB. avail mem=23.00 GB. ⏎ [2025-07-22 06:18:34 TP0] max_total_num_tokens=6004242, chunked_prefill_size=16384, max_prefill_tokens=16384, max_running_requests=4096, context_len=40960, available_gpu_mem=23.00 GB ⏎ [2025-07-22 06:18:35] INFO:     Started server process [163598] ⏎ [2025-07-22 06:18:35] INFO:     Waiting for application startup. ⏎ [2025-07-22 06:18:35] INFO:     Application startup complete. ⏎ [2025-07-22 06:18:35] INFO:     Uvicorn running on http://127.0.0.1:30000 (Press CTRL+C to quit) ⏎ [2025-07-22 06:18:36] INFO:     127.0.0.1:42888 - "GET /get_model_inf …[truncated]

### L1-2262369905  (L1, 2025-07-28, sha 226236990588, PR #8457)
TITLE: Revert "[kernel] opt moe align block kernel by block/warp scan algorithm" (#8457)
SOURCES: path_core, subject_keyword, release_notes, corpus:confirmed-reverts
STAGE1: revert; artifacts=L1.align.cuda_aot; Reverts prior moe_align block/warp scan optimization.
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/csrc/moe/moe_align_kernel.cu (+42/-51)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 7884 reason=hardware_specific_breakage
BODY: Reverts sgl-project/sglang#7884 ⏎  ⏎ PR has bug, and it caused a lot of test failture in ci.

### L1-134fa43e19  (L1, 2025-07-28, sha 134fa43e1940, PR #8453)
TITLE: [NVIDIA] Change to use `num_local_experts` (#8453)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.ep.layer,L1.upstream.flashinfer_moe; Changes EPMoE to use num_local_experts, fixing trtllm-moe-fp8 usage.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-1); docs/backend/server_arguments.md (+2/-1)
BODY: The recent [change](https://github.com/sgl-project/sglang/commit/bf0f448fe5b549cc80bc86a505e0ceb040e0f613) breaks the usage of trtllm-moe-fp8 kernel. This PR fixes it. ⏎  ⏎ cc @kushanam @zhyncs

### L1-9c138a0445  (L1, 2025-07-28, sha 9c138a044514, PR #8421)
TITLE: [3/N] MoE Refactor: Simplify DeepEP Output (#8421)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: adapt_framework; artifacts=L1.ep.deepep_dispatcher,L1.ep.layer; Introduces DispatchOutput and simplifies DeepEP dispatcher output protocol.
ARTIFACT_HINTS: L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+150/-30); python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+69/-118); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+0/-0); python/sglang/srt/layers/moe/token_dispatcher/base_dispatcher.py (+48/-0); python/sglang/srt/layers/moe/token_dispatcher/standard.py (+19/-0); python/sglang/srt/models/deepseek_v2.py (+13/-56); python/sglang/srt/models/qwen3_moe.py (+12/-69); python/sglang/srt/two_batch_overlap.py (+8/-3)
BODY: - Introduce `DispatchOutput` to maintain dispatcher's results. ⏎ - Move DeepEP's `dispatch` and `combine` operations from model files the moe layer file. ⏎  ⏎ After this PR, all forward functions of MoE share the same logic: dispatch -> layout transfer -> grouped-gemm. ⏎  ⏎ ## Checklist

### L1-74e7e45710  (L1, 2025-07-28, sha 74e7e457103a, PR #8469)
TITLE: Fix DEEPEP BF16 compatibility for Deepseek Style model like GLM 4.5 (#8469)
SOURCES: path_core, subject_keyword, release_notes, corpus:kernel-correctness-cases
STAGE1: repair_correctness; artifacts=L1.ep.layer; Fixes DeepEP BF16 compatibility for DeepSeek-style MoE models.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-6)
DEEP_STUDY: deep-study correctness case sglang:74e7e45710: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: ## Motivation ⏎ Fix DEEPEP BF16 compatibility surfaced by running ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model /shared/public/elr-models/GLM-4.5/ --tp-size 8 --trust-remote-code --enable-deepep-moe --deepep-mode=normal ⏎ ``` ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-a9dd3ec3e9  (L1, 2025-07-28, sha a9dd3ec3e961, PR #8125)
TITLE: fix:reorder topk experts to ensure shared expert replaces minimal score (#8125)
SOURCES: path_core, symbol_pickaxe
STAGE1: repair_correctness; artifacts=L1.routing.topk_py; Reorders top-k experts so shared expert replaces the minimal-score expert.
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+6/-2)
DEEP_STUDY: deep-study correctness case sglang:a9dd3ec3e9: class=other; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: ## Motivation ⏎  ⏎ In the MoE (Mixture of Experts) layer, in the original design, to achieve integration between the shared expert and routed experts, when selecting top-k experts for each token, we need to reorder the selected experts to guarantee the last expert in the list has the smallest score. The original implementation uses torch.topk with sorted=False, which means the returned experts aren't necessarily in score order. When we replace the last expert with the shared expert, we might accidentally replace a unexpected expert, altering the layer's expected behavior. This fix ensures proper expert selection by sorting the top-k experts before replacement. ⏎  ⏎ ## Modifications ⏎  ⏎ Before: Used raw torch.topk(..., sorted=False) output for expert replacement ⏎  ⏎ After: Now sorts selected experts by score before shared expert substitution ⏎  ⏎ ## Checklist

### L1-4d16c88b6e  (L1, 2025-07-29, sha 4d16c88b6e96, PR #8535)
TITLE: Update cutlass_moe.py (#8535)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.cutlass.adapters; Fixes cutlass_moe return path for apply_shuffle_mul_sum result.
ARTIFACT_HINTS: L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+2/-1)
BODY: Minor change regarding cutlass MoE ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎ Fix a minor bug ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-9effeb5bdd  (L1, 2025-07-29, sha 9effeb5bddf2, PR #8448)
TITLE: Support EPLB in FusedMoE (#8448)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: extend_support; artifacts=L1.triton.fused_moe,L1.ep.layer; Adds EPLB expert mapping support into FusedMoE/EPMoE execution paths.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer
FILES: python/sglang/srt/eplb/expert_distribution.py (+5/-0); python/sglang/srt/eplb/expert_location.py (+17/-6); python/sglang/srt/eplb/expert_location_dispatch.py (+1/-0); python/sglang/srt/eplb/expert_location_updater.py (+2/-0); python/sglang/srt/layers/moe/ep_moe/layer.py (+16/-3); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+44/-1); python/sglang/srt/models/deepseek_v2.py (+2/-0); python/sglang/srt/models/glm4_moe.py (+3/-1); python/sglang/srt/models/grok.py (+3/-0); python/sglang/srt/models/llama4.py (+3/-0); python/sglang/srt/models/mixtral.py (+3/-0); python/sglang/srt/models/granitemoe.py (+3/-0); python/sglang/srt/models/hunyuan.py (+1/-0); python/sglang/srt/models/olmoe.py (+3/-0); python/sglang/srt/models/phimoe.py (+1/-0)
BODY: ## Motivation ⏎  ⏎ Fix #8398 ⏎  ⏎ ## Modifications ⏎  ⏎ ## Checklist

### L1-e3f08c77bc  (L1, 2025-07-29, sha e3f08c77bc8e, PR #8545)
TITLE: Update cutlass_moe.py (#8545)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.cutlass.adapters; Casts topk_weights to output dtype before cutlass_moe combine.
ARTIFACT_HINTS: L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+1/-1)
BODY: Another small fix for cutlass moe ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎ Simple bug fix for FP8 cutlass moe ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-9b9e82539b  (L1, 2025-07-30, sha 9b9e82539b77, PR #8564)
TITLE: [Fix]Fix index oob in get_group_gemm_starts kernel. (#8564)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.cutlass.fp8_blockwise; Fixes int offset/stride OOB in CUTLASS grouped-GEMM starts helper.
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise
FILES: sgl-kernel/csrc/moe/cutlass_moe_helper.cu (+6/-6)
BODY: ## Motivation ⏎ Using `int` as offset or stride may sometimes cause OOB problems. For example, when M=128, N=8192, K=8192, and groups=256, executing the following code will cause OOB:https://github.com/sgl-project/sglang/blob/55ecdc0a8e62ac56bb475f128d2b1fc728953a28/sgl-kernel/csrc/moe/cutlass_moe_helper.cu#L56 ⏎ So I fix it. ⏎  ⏎  ⏎ ## Modifications ⏎ sgl-kernel/csrc/moe/cutlass_moe_helper.cu ⏎  ⏎  ⏎ ## Accuracy Test ⏎ Test M=128, N=8192, K=8192, groups=256: ⏎ <img width="3836" height="480" alt="image" src="https://github.com/user-attachments/assets/fa4646e9-0693-4fb2-807f-689e2ebf5d80" /> ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-a5f5ab4030  (L1, 2025-07-30, sha a5f5ab4030a1, PR #8514)
TITLE: update sgl-kernel for EP: kernel part  (#8514)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L1.align.cuda_aot; Changes moe_align kernel contract to accept -1 expert IDs from EP filtering.
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/csrc/moe/moe_align_kernel.cu (+6/-7); sgl-kernel/python/sgl_kernel/moe.py (+0/-2); sgl-kernel/benchmark/bench_moe_align_block_size.py (+0/-10); sgl-kernel/csrc/common_extension.cc (+1/-1); sgl-kernel/csrc/torch_extension_rocm.cc (+1/-1); sgl-kernel/include/sgl_kernel_ops.h (+0/-1); sgl-kernel/tests/test_moe_align.py (+4/-10)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ In EP, we set the expert ids for filtered experts as -1. We update sgl-kernel to handle this case. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-e179e0b797  (L1, 2025-07-31, sha e179e0b79738, PR #8550)
TITLE: update sgl-kernel for EP: python part (#8550)
SOURCES: path_core, dependency_pin
STAGE1: adapt_framework; artifacts=L1.triton.fused_moe,L1.align.cuda_aot; Updates Python fused_moe side for EP filtered expert IDs and sgl-kernel dependency.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+4/-9); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎ See #8514  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-3bdcdd134b  (L1, 2025-07-31, sha 3bdcdd134b1c, PR #8461)
TITLE: [Hot-Fix] moe_aligned_block_size CI failed in AMD (#8461)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: reland; artifacts=L1.align.cuda_aot; Relands moe_align improvement with separate HIP/CUDA branches after AMD CI failure.
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/csrc/moe/moe_align_kernel.cu (+65/-6)
BODY: ## Motivation ⏎ This PR is to re-introduce the improvement of https://github.com/sgl-project/sglang/pull/7884, which was reverted in https://github.com/sgl-project/sglang/pull/8457 due to AMD CI failed constantly. ⏎  ⏎ This PR is to diverge the branches for HIP and CUDA. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-32fa1e9cc2  (L1, 2025-07-31, sha 32fa1e9cc286, PR #8515)
TITLE: [4/N] MoE Refactor: Unified Triton Kernel for FusedMoE and EPMoE (#8515)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: adapt_framework; artifacts=L1.triton.fused_moe,L1.ep.layer; Unifies Triton kernel path for FusedMoE and EPMoE, changing adapter structure.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+15/-648); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+22/-4); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+32/-12); python/sglang/srt/layers/quantization/fp8.py (+0/-18); python/sglang/srt/layers/quantization/unquant.py (+0/-8); python/sglang/srt/layers/quantization/w4afp8.py (+1/-0)
ISSUES: #8402 [Bug] DeepSeek-V3 model gets bad accuracy result on gsm8k benchmark when EP is enabled | #8427 [Feature] Support per channel quant EPMOE Triton kernel
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-7a1f7fc504  (L1, 2025-07-31, sha 7a1f7fc5049d, PR #8590)
TITLE: [Feature] Hybrid EP and TP (#8590)
SOURCES: path_core, symbol_pickaxe
STAGE1: extend_support; artifacts=L1.triton.fused_moe,L1.ep.layer; Adds hybrid EP/TP support touching FusedMoE and EPMoE layer behavior.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+21/-25); assets/logo.svg (+1/-1); assets/logo_square.svg (+1/-1); python/sglang/bench_one_batch.py (+3/-0); python/sglang/srt/distributed/parallel_state.py (+86/-1); python/sglang/srt/entrypoints/engine.py (+2/-0); python/sglang/srt/managers/data_parallel_controller.py (+2/-0); python/sglang/srt/managers/scheduler.py (+11/-1); python/sglang/srt/managers/tp_worker.py (+4/-0); python/sglang/srt/managers/tp_worker_overlap_thread.py (+2/-1); python/sglang/srt/model_executor/model_runner.py (+5/-0); python/sglang/srt/server_args.py (+1/-8); python/sglang/srt/speculative/eagle_worker.py (+2/-0)
PERF_LINES: Accuracy: 93.5%
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Dependency: ⏎ - #8514  ⏎ - #8515  ⏎ - #8550  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path /dev/shm/GLM-4.5-Air-FP8 --trust-remote-code --tp 4 --enable-ep-moe --base-gpu-id 4 --ep-size 2 ⏎ python3 few_shot_gsm8k.py ⏎ ``` ⏎  ⏎ Accuracy: 93.5% ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b7170cc820  (L1, 2025-07-31, sha b7170cc82062, PR #8630)
TITLE: [bugfix] Fix flashinfer cutlass EP moe after MoE refactor (#8630)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: repair_correctness; artifacts=L1.runner.flashinfer_cutlass,L1.triton.fused_moe; Fixes FlashInfer CUTLASS EP MoE accuracy after MoE refactor.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-1); python/sglang/srt/server_args.py (+5/-0)
LABELS: bug, high priority
PERF_LINES: Latency: 23.725 s | Output throughput: 6143.771 token/s
BODY: ## Motivation ⏎  ⏎ Accuracy went to 0 for flashinfer cutlass MoE after recent MoE refactor and improvements. ⏎  ⏎ ## Modifications ⏎  ⏎ * Fix issue from https://github.com/sgl-project/sglang/pull/8515 - cutlass moe does not need to map topkid ids using expert map gpu. ⏎ * Fix issue from https://github.com/sgl-project/sglang/pull/8590 - hybrid EP/TP not yet supported for this path ⏎  ⏎ ## Accuracy Test ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path nvidia/DeepSeek-R1-0528-FP4 --trust-remote-code --quantization modelopt_fp4 --tp 8 --enable-flashinfer-cutlass-moe --enable-ep-moe --ep-size 8 ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319 --port=30000 ⏎ Accuracy: 0.961 ⏎ Invalid: 0.000 ⏎ Latency: 23.725 s ⏎ Output throughput: 6143.771 token/s ⏎ ``` ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-aa4c66b564  (L1, 2025-07-31, sha aa4c66b564b7, PR #8450)
TITLE: [NVIDIA] Enable Flashinfer MoE blockscale fp8 backend for TP MoE (#8450)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: integrate; artifacts=L1.upstream.flashinfer_moe,L1.triton.fused_moe; Enables FlashInfer MoE blockscale FP8 backend for TP MoE path.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+19/-34); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+54/-1); python/sglang/srt/layers/quantization/fp8.py (+52/-0); python/sglang/srt/models/deepseek_v2.py (+3/-4); python/sglang/srt/models/glm4_moe.py (+3/-3); python/sglang/srt/server_args.py (+0/-4)
LABELS: high priority
BODY: A followup PR to enable Flashinfer MoE blockscale fp8 backend for TP MoE. ⏎  ⏎ The previous [PR](https://github.com/sgl-project/sglang/pull/8036) is doing the same but for the EP MoE. ⏎  ⏎ cc. @kushanam

### L1-c8d3a402c1  (L1, 2025-08-01, sha c8d3a402c1ca, PR #8511)
TITLE: Bug: apply final_hidden_states*=self.routed_scaling_factor at MoE lay… (#8511)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: repair_correctness; artifacts=L1.ep.layer; Applies routed_scaling_factor to EPMoE final_hidden_states to fix accuracy regression.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-1)
BODY: …er if epmoe is enabled ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎ Issue: https://github.com/sgl-project/sglang/issues/8402 ⏎  ⏎ Also noticed significant regression when running epmoe during recent GLM4.5 support work: GSM8K accuracy drops from 0.965 to 0.745 when EPMOE is enabled. Accuracy is good for TP & DeepEP. ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model  /shared/public/elr-models/zai-org/GLM-4.5 --tp-size 8 --trust-remote-code ⏎  ⏎ python3 benchmark/gsm8k/bench_sglang.py ⏎ ``` ⏎  ⏎ ## Modifications ⏎  ⏎ There is a bug in DeepSeek-V2 and GLM-4.5 related to how routed_scaling_factor is applied in MoE (Mixture-of-Experts) layers. ⏎  ⏎ Currently, the routed_scaling_factor is applied in three different places, leading to ambiguity: ⏎ - self.topk.forward: The scaling is applied only in the n out of m condition, but not in the m - n out of m condition. ⏎ - self.experts.forward: Same as above — the factor is only applied in n out of m. ⏎ - Model-level logic: The model itself may apply routed_scaling_factor, but it cannot know whether self.topk or self.experts have already done so. ⏎  ⏎ This results in uncertain and inconsistent scaling, as the model layer has no visibility into whether routed_scaling_factor has already been applied upstream. ⏎  ⏎ 🎪 TL;DR: The model can't know if it should apply * routed_scaling_factor or not, because topk and experts may or may not have already done it, depending on the code path. ⏎  ⏎ My PR forces model layer to apply * routed_scaling_factor when EPMOE is enabled because from the current codebase, epmoe won't apply *routed_scaling_factor by itself.  ⏎  ⏎ Need to follow up and possibly refactor sglang moe codebase to make it clear which layer should apply * routed_scaling_factor  ⏎  ⏎ ## Accuracy Test ⏎ - GLM4.5 GSM8K accuracy jumps from 0.745 to 0.965 under EPMOE which is the same as TP & DeepEP results ⏎ - Need to run gsm8k for deepseek-v3 too - WIP ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-6c88f6c8d9  (L1, 2025-08-01, sha 6c88f6c8d908, PR #8658)
TITLE: [5/N] MoE Refactor: Update MoE parallelism arguments (#8658)
SOURCES: path_core, symbol_pickaxe, corpus:production-kernel-provenance
STAGE1: adapt_framework; artifacts=L1.ep.deepep_dispatcher,L1.ep.layer; Introduces moe-a2a-backend argument and updates DeepEP dispatcher registration/protocol.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+9/-35); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-5); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+23/-0); python/sglang/srt/layers/moe/token_dispatcher/base_dispatcher.py (+12/-1); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+8/-15); python/sglang/srt/layers/moe/utils.py (+43/-0); docker/k8s-sglang-distributed-sts.yaml (+1/-2); docs/backend/pd_disaggregation.md (+8/-8); docs/backend/server_arguments.md (+1/-2); docs/references/disaggregation/lws-examples/d.yaml (+4/-6); docs/references/disaggregation/lws-examples/p.yaml (+4/-6); docs/references/disaggregation/lws_pd_deploy.md (+8/-12); python/sglang/srt/eplb/expert_distribution.py (+4/-2); python/sglang/srt/layers/communicator.py (+1/-1); python/sglang/srt/managers/schedule_batch.py (+2/-2); python/sglang/srt/managers/scheduler.py (+5/-3); python/sglang/srt/model_executor/forward_batch_info.py (+2/-1); python/sglang/srt/model_executor/model_runner.py (+5/-0); python/sglang/srt/models/deepseek_v2.py (+10/-15); python/sglang/srt/models/glm4_moe.py (+10/-15); python/sglang/srt/models/grok.py (+3/-3); python/sglang/srt/models/mixtral.py (+3/-3); python/sglang/srt/models/qwen2_moe.py (+1/-4); python/sglang/srt/models/qwen3_moe.py (+7/-8); python/sglang/srt/models/step3_vl.py (+1/-1); python/sglang/srt/operations_strategy.py (+1/-1); python/sglang/srt/server_args.py (+47/-20); python/sglang/srt/two_batch_overlap.py (+5/-4); python/sglang/srt/utils.py (+2/-23); python/sglang/test/runners.py (+0/-2); test/srt/test_deepep_large.py (+4/-2); test/srt/test_deepep_small.py (+14/-7); test/srt/test_eplb.py (+3/-3); test/srt/test_hybrid_dp_ep_tp_mtp.py (+80/-80); test/srt/test_moe_deepep.py (+4/-2); test/srt/test_moe_deepep_eval_accuracy_large.py (+2/-1); test/srt/test_moe_ep.py (+0/-2); test/srt/test_two_batch_overlap.py (+4/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR introduces `--moe-a2a-backend` and deprecates `--enable-ep-moe` and `--enable-deepep-moe`. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-1fe691a429  (L1, 2025-08-01, sha 1fe691a429be, PR #8648)
TITLE: Fix FP8 block quantization when N or K is not multiples of 128 (#8648)
SOURCES: path_core, corpus:kernel-correctness-cases
STAGE1: repair_correctness; artifacts=L1.hardware.cpu_npu_musa; Fixes CPU MoE FP8 block quantization for N/K not multiples of 128.
ARTIFACT_HINTS: -
FILES: sgl-kernel/csrc/cpu/moe.cpp (+10/-10); test/srt/cpu/test_moe.py (+10/-4); test/srt/cpu/utils.py (+19/-4)
LABELS: ready-to-merge
DEEP_STUDY: deep-study correctness case sglang:1fe691a429: class=shape_alignment_edge; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ This PR is to add support of FP8 block quantize when N or K is not multiples of 128. ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-f642524fd9  (L1, 2025-08-01, sha f642524fd992, PR #8364)
TITLE: [1/2] sgl-kernel: Fuse routed scaling factor into select_experts (#8364)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:confirmed-reverts(reverted)
STAGE1: optimize; artifacts=L1.routing.fused_gate; Fuses routed_scaling_factor multiplication into moe_fused_gate/select_experts kernel.
ARTIFACT_HINTS: L1.routing.fused_gate
FILES: sgl-kernel/csrc/common_extension.cc (+1/-1); sgl-kernel/csrc/moe/moe_fused_gate.cu (+20/-7); sgl-kernel/include/sgl_kernel_ops.h (+2/-1); sgl-kernel/python/sgl_kernel/moe.py (+9/-2); sgl-kernel/tests/test_moe_fused_gate.py (+6/-1)
LABELS: high priority
DEEP_STUDY: deep-study: this PR was reverted by PR 8706 (confirmed_revert, reason=unstated)
PERF_LINES: 10.46% speedup at BS 1 | 1.86% speedup at BS 128 | 1.22% speedup at BS1 | 0.26% speedup at BS128 | latency: 10.60 s | last generation throughput: 96.61 tok/s | input throughput: 9392.46 tok/s | output throughput: 97.62 tok/s | latency: 39.79 s | last generation throughput: 3693.98 tok/s | input throughput: 17344.15 tok/s | output throughput: 4066.24 tok/s | latency: 10.46 s | last generation throu
BODY: ## Motivation ⏎  ⏎ Follow up to https://github.com/sgl-project/sglang/pull/8333 ⏎ Fuse the multiply by routed_scaling_factor into select_experts, following example of TRT-LLM: ⏎ https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/models/modeling_deepseekv3.py#L323 ⏎ https://github.com/NVIDIA/TensorRT-LLM/blob/738ab615930fd08dccb94fa388bd74dc91c5f235/cpp/tensorrt_llm/kernels/noAuxTcKernels.cu#L651 ⏎  ⏎ For the non-FP4 paths, the routed_scaling_factor is fused into moe_sum_reduce. However, we could move it into select_experts for those paths too if we wanted to simplify the code. ⏎  ⏎ ## Modifications ⏎  ⏎ Add boolean argument to fused_moe_gate() ⏎  ⏎  ⏎ ## Results ⏎  ⏎ Prefill: ⏎ 10.46% speedup at BS 1 ⏎ 1.86% speedup at BS 128 ⏎ Decode: ⏎ 1.22% speedup at BS1 ⏎ 0.26% speedup at BS128 ⏎  ⏎ Server command ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path nvidia/DeepSeek-R1-0528-FP4 --trust-remote-code --quantization modelopt_fp4 --tp 8 --enable-flashinfer-cutlass-moe --enable-ep-moe --ep-size 8 ⏎ ``` ⏎  ⏎ BEFORE results ⏎ ``` ⏎ python3 -m sglang.bench_one_batch_server --model-path nvidia/DeepSeek-R1-0528-FP4 --base-url http://127.0.0.1:30000/ --batch-size 1 --input-len 1024 --output-len 1024 ⏎  ⏎ #Input tokens: 1024 ⏎ #Output tokens: 1024 ⏎ batch size: 1 ⏎ input_len: 1024 ⏎ output_len: 1024 ⏎ latency: 10.60 s ⏎ ttft: 0.11 s ⏎ last generation throughput: 96.61 tok/s ⏎ input throughput: 9392.46 tok/s ⏎ output throughput: 97.62 tok/s ⏎  ⏎ python3 -m sglang.bench_one_batch_server --model-path nvidia/DeepSeek-R1-0528-FP4 --base-url http://127.0.0.1:30000/ --batch-size 128 --input-len 1024 --output-len 1024 ⏎  ⏎ #Input tokens: 131072 ⏎ #Output tokens: 131072 ⏎ batch size: 128 ⏎ input_len: 1024 ⏎ output_len: 1024 ⏎ latency: 39.79 s ⏎ ttft: 7.56 s ⏎ last generation throughput: 3693.98 tok/s ⏎ input throughput: 17344.15 tok/s ⏎ output throughput: 4066.24 tok/s ⏎ ``` ⏎  ⏎ AFTER results ⏎ ``` ⏎ python3 -m sglang.bench_one_batch_server --model-path nvidia/DeepSeek-R1-0528-FP4 --base-url http://127.0.0.1:30000/ --batch-size 1 --input-len 1024 --output-len 1024 ⏎  ⏎ #Input tokens: 1024 ⏎ #Output tokens: 1024 ⏎ batch size: 1 ⏎ input_len: 1024 ⏎ output_len: 1024 ⏎ latency: 10.46 s ⏎ ttft: 0.10 s ⏎ last generation throughput: 97.88 tok/s ⏎ input throughput: 10375.70 tok/s ⏎ output throughput: 98.82 tok/s ⏎  ⏎ python3 -m sglang.bench_one_batch_server --model-path nvidia/DeepSeek-R1-0528-FP4 --base-url http://127.0.0.1:30000/ --batch-size 128 --input-len 1024 --output-len 1024 ⏎  ⏎ #Input tokens: 131072 ⏎ #Output tokens: 131072 ⏎ batch size: 128 ⏎ input_len: 1024 ⏎ output_len: 1024 ⏎ latency: 39.57 s ⏎ ttft: 7.42 s ⏎ last generation t …[truncated]

### L1-89caf7a3c6  (L1, 2025-08-01, sha 89caf7a3c6cd, PR #8688)
TITLE: [bugfix] Apply routed scaling factor to cutlass_fused_experts_fp8 (#8688)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: repair_correctness; artifacts=L1.cutlass.adapters; Applies routed_scaling_factor to cutlass_fused_experts_fp8 output for accuracy.
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/fp8.py (+5/-1)
BODY: ## Motivation ⏎  ⏎ Similar to https://github.com/sgl-project/sglang/pull/8333 we weren't applyign routed scaling factor for cutlass_fused_experts_fp8 (SGLANG_CUTLASS_MOE=1) path. ⏎  ⏎ https://github.com/sgl-project/sglang/pull/8364 will fuse this into select_experts, but let's get the accuracy fixed first since that requires an sgl-kernel change and won't be merged for a while. ⏎  ⏎ ## Modifications ⏎  ⏎ Multiply MOE output by factor. ⏎  ⏎ ## Accuracy Test ⏎  ⏎ `` ⏎ SGL_ENABLE_JIT_DEEPGEMM=0 SGLANG_CUTLASS_MOE=1 SGLANG_ENABLE_FLASHINFER_GEMM=1 python3 -m sglang.launch_server --tp=8 --trust-remote-code --disable-radix-cache --model-path=deepseek-ai/DeepSeek-R1-0528 --attention-backend=cutlass_mla ⏎ ``` ⏎ SGL_CUTLASS_MOE=1 FP8 path was giving 0.832 accuracy, now it gives 0.965. ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-8ada1ab6c7  (L1, 2025-08-02, sha 8ada1ab6c791, PR #8705)
TITLE: Fix triton moe error caused by TopK refactor (#8705)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.runner.openai_triton_kernels; Fixes Triton-kernels MoE adapter after TopK refactor.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.openai_triton_kernels, L1.upstream.openai_triton_kernels
FILES: python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py (+0/-31)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-f9f0138f80  (L1, 2025-08-02, sha f9f0138f80a3, PR #8706)
TITLE: Revert "[1/2] sgl-kernel: Fuse routed scaling factor into select_experts" (#8706)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:confirmed-reverts
STAGE1: revert; artifacts=L1.routing.fused_gate; Reverts routed_scaling_factor fusion in moe_fused_gate/select_experts.
ARTIFACT_HINTS: L1.routing.fused_gate
FILES: sgl-kernel/csrc/common_extension.cc (+1/-1); sgl-kernel/csrc/moe/moe_fused_gate.cu (+7/-20); sgl-kernel/include/sgl_kernel_ops.h (+1/-2); sgl-kernel/python/sgl_kernel/moe.py (+2/-9); sgl-kernel/tests/test_moe_fused_gate.py (+1/-6)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 8364 reason=unstated
BODY: Reverts sgl-project/sglang#8364

### L1-d9def43dcd  (L1, 2025-08-02, sha d9def43dcdfd, PR #8722)
TITLE: [Perf]Use Cooperative Schedule for H100 & H200 & H800 in fp8_blockwise_scaled_grouped_mm (#8722)
SOURCES: path_core
STAGE1: optimize; artifacts=L1.cutlass.fp8_blockwise; Switches CUTLASS FP8 blockwise grouped GEMM schedule for H100/H200/H800 performance.
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise
FILES: sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu (+3/-2)
PERF_LINES: 2. The fp8 blockwise calculation logic is to execute 4 Tensor Core MMAs and then 1 CUDA Core FMA. When migrating from H20 to H100, H200, and H800, the time for  | Clearly, the Cooperative Schedule's competition for Tensor Cores can address the issue of declining Tensor Pipe Throughput. Furthermore, Cooperative Epilogue of
BODY: ## Motivation ⏎ Fix https://github.com/sgl-project/sglang/pull/7278#discussion_r2225518835 ⏎ Migrating fp8_blockwise_scaled_grouped_mm from H20 to H100, H200, and H800 resulted in a performance regression. After in-depth profiling, we identified several key causes of the regression: ⏎  ⏎ 1. The reduction in Mainloop execution time may not be enough to cover the Epilogue overhead. ⏎ 2. The fp8 blockwise calculation logic is to execute 4 Tensor Core MMAs and then 1 CUDA Core FMA. When migrating from H20 to H100, H200, and H800, the time for the 4 Tensor Core MMAs is obviously reduced to 1/4 of the original time. In the Pingpong schedule, the mainloops of the two warp groups are interleaved, which means that CUDA Core FMA cannot be overlapped. Originally, its proportion on H20 was low, but now after migrating to H100, H200, and H800, its proportion has increased significantly, resulting in a decrease in SM Tensor Pipe Throughput. ⏎ 3. In the Pingpong schedule, the TMA copies smaller tiles and completes the input matrix copying with more requests. This approach is less efficient and causes more Stall Long Scoreboards. ⏎  ⏎ Clearly, the Cooperative Schedule's competition for Tensor Cores can address the issue of declining Tensor Pipe Throughput. Furthermore, Cooperative Epilogue offers pipeline optimizations, which is why we chose to use Cooperative Schedule on the H100, H200, and H800. ⏎  ⏎ Thanks @yuan-luo for reporting the performance regression. ⏎  ⏎  ⏎ ## Modifications ⏎ sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu ⏎  ⏎  ⏎ ## Accuracy Test ⏎ <img width="3448" height="374" alt="image" src="https://github.com/user-attachments/assets/79d73652-16e4-43af-aa27-c82154721606" /> ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling(Running on H200) ⏎ Before: ⏎ <img width="2116" height="486" alt="H200 Baseline" src="https://github.com/user-attachments/assets/5546b4df-07a6-48f5-b00f-cbc2645221b0" /> ⏎ After: ⏎ <img width="2116" height="442" alt="H200 OPT" src="https://github.com/user-attachments/assets/f35031f8-7edc-4893-81c8-ee045861b5d0" /> ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-3435a24e81  (L1, 2025-08-03, sha 3435a24e8157, PR #8676)
TITLE: [RL] fix update weight for FusedMoE with EP (#8676)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:kernel-correctness-cases
STAGE1: repair_correctness; artifacts=L1.triton.fused_moe; Fixes FusedMoE weight update handling under EP.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+11/-3)
LABELS: high priority
DEEP_STUDY: deep-study correctness case sglang:3435a24e81: class=integration_backend_cudagraph; symptom=performance_or_availability; introducing=unknown
BODY: ## Motivation ⏎  ⏎  ⏎ During colocated RL, we will use `/release_memory_occupation` to release all GPU allocations within model init, and we should not create GPU tensors that cannot be loaded through weight_loader. ⏎  ⏎ This PR moves the creation of `self.expert_map_gpu` to `forward` and make sure `self.expert_map_cpu` is created on CPU (which was by default created on GPU). ⏎  ⏎ Thank you for your time on reviewing this PR :) ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist
