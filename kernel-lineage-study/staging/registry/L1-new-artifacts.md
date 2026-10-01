# L1 NEW artifact proposals from stage-2 coding (42 names)

## NEW:moe_output_reduction  (14 events)
- L1-1e8699fda3 2026-09-18 PR #32963: [NVIDIA][comm] Merge EP+MoE-TP post-experts all-reduces into one _TP reduction (#32963) | paths: python/sglang/srt/layers/moe/__init__.py; python/sglang/srt/layers/moe/utils.py | notes: New artifact not in registry.
- L1-62ae032c46 2026-09-24 PR #41097: [Refactor] Share the MoE output all-reduce between models (#41097) | paths: python/sglang/srt/layers/moe/__init__.py; python/sglang/srt/layers/moe/utils.py | notes: NEW artifact is the shared MoE output-reduction utility/protocol.
- L1-402df23188 2026-09-25 PR #41195: [Fix] Stop counting a deferred FFN sum more than once: replicated TP1 shared expert, dense reduce_sc | paths: python/sglang/srt/layers/moe/utils.py | notes: 
- L1-9d7f44bdbb 2026-09-25 PR #41196: [Refactor] Carry a deferred FFN all-reduce as UnreducedOutput and complete it in the next layer with | paths: python/sglang/srt/layers/moe/__init__.py; python/sglang/srt/layers/moe/cutedsl_ar_fusion.py | notes: 
- L1-3450d68d4e 2026-09-25 PR #41200: [Refactor] Decide an FFN exit's completion once and declare the group it owes (#41200) | paths: python/sglang/srt/layers/moe/__init__.py; python/sglang/srt/layers/moe/utils.py | notes: 
- L1-8fc3ce48da 2026-09-26 PR #41252: [Refactor] Choose prepare_attn / prepare_mlp steps and fused kernels at construction (#41252) | paths: python/sglang/srt/layers/moe/cutedsl_ar_fusion.py | notes: 

## NEW:cutedsl_ar_fusion  (5 events)
- L1-3177d10ca6 2026-09-24 PR #39816: Refactor the Cute-DSL AR fusion to support DeepseekV2 archs (GLM-5.3, etc.) (#39816) | paths: python/sglang/srt/layers/moe/cutedsl_ar_fusion.py; python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py; python/sglang/srt/layers/moe/qwen35_flashinfer_fusion.py | notes: NEW artifact is the generic CuTe-DSL all-reduce/MoE-finalize fusion layer.
- L1-8fc3ce48da 2026-09-26 PR #41252: [Refactor] Choose prepare_attn / prepare_mlp steps and fused kernels at construction (#41252) | paths: python/sglang/srt/layers/moe/cutedsl_ar_fusion.py | notes: 
- L1-846181f1c4 2026-09-27 PR #41418: [Refactor] Give the fused prepare_mlp kernels an explicit contract (#41418) | paths: python/sglang/srt/layers/moe/cutedsl_ar_fusion.py | notes: 
- L1-f0ecf15c84 2026-09-27 PR #41431: [Refactor] Move CuTe DSL-fused layers onto the declared boundaries and FFN-exit kernel entries (#414 | paths: python/sglang/srt/layers/moe/cutedsl_ar_fusion.py | notes: 
- L1-06d012eaca 2026-09-27 PR #41442: [Refactor] Give a layer its CuTe DSL kernels at construction instead of a subclass (#41442) | paths: python/sglang/srt/layers/moe/cutedsl_ar_fusion.py | notes: 

## NEW:moe_lora_align_jit  (2 events)
- L1-af2807e146 2026-03-12 PR #19710: [LoRA][I] Add MOE LoRA JIT alignment kernel and tests  (#19710) | paths: python/sglang/jit_kernel/csrc/lora/moe_lora_align_kernel.cu; python/sglang/jit_kernel/moe_lora_align.py | notes: New artifact because LoRA-specific JIT alignment wrapper is not in the L1 registry.
- L1-d3b972cbf0 2026-08-28 PR #36379: fix(lora): build the MoE LoRA align JIT kernel on ROCm (#36379) | paths: python/sglang/kernels/jit/csrc/lora/moe_lora_align_kernel.cu | notes: 

## NEW:routed_experts_capturer  (2 events)
- L1-4a279d9c36 2026-05-06 PR #24550: [R3] Avoid implicit CUDA sync in routed experts DP slicing (#24550) | paths:  | notes: NEW artifact is routed-experts DP state capturer.
- L1-376635c1e3 2026-05-30 PR #26123: Fix routed-experts device buffer overflow under DP attention (#26123) | paths:  | notes: NEW artifact is a state capturer, not in L1 registry.

## NEW:moe_finalize_fuse_shared  (2 events)
- L1-d1a39b0c74 2026-06-12 PR #27720: [DeepSeek V3] Defer moe finalize and fused it with main stream add (#27720) | paths: python/sglang/jit_kernel/csrc/moe/moe_finalize_fuse_shared.cu; python/sglang/jit_kernel/csrc/moe/tvm_ffi_utils.h; python/sglang/jit_kernel/moe_finalize_fuse_shared.py; python/sglang/srt/layers/moe/fused_moe_triton/layer.py | notes: NEW artifact is the new JIT moe_finalize_fuse_shared kernel.
- L1-72d5c5bb73 2026-09-09 PR #38612: [Kimi-K3] Accept fp32 routing weights in the fused MoE finalize (#38612) | paths: python/sglang/kernels/jit/csrc/moe/moe_finalize_fuse_shared.cu; python/sglang/srt/models/kimi_k3.py | notes: New artifact not in registry.

## NEW:humming_moe_runner  (2 events)
- L1-423b8485fb 2026-07-14 PR #23754: [Quantization] add humming quantization kernel (#23754) | paths: python/sglang/jit_kernel/csrc/moe/moe_permute_prepare.cu; python/sglang/jit_kernel/moe_permute_prepare.py; python/sglang/kernels/ops/moe/__init__.py; python/sglang/srt/layers/moe/ep_moe/kernels.py | notes: NEW:humming_moe_runner is not in registry; PR introduces a new runner and kernels.
- L1-21258b7a35 2026-08-24 PR #31429: feat(humming): FP8 DeepEP dispatch for humming MoE backend (#31429) | paths: python/sglang/kernels/ops/moe/ep_moe_kernels.py; python/sglang/srt/layers/moe/moe_runner/humming.py; python/pyproject.toml; python/sglang/srt/layers/quantization/humming.py | notes: 

## NEW:inkling_moe  (2 events)
- L1-02cd44c59a 2026-08-05 PR #33108: feat(dgx-spark): add inkling-small MoE support for sm_121 (#33108) | paths: python/sglang/kernels/ops/moe/inkling_moe.py; python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/silu_and_mul_interleaved_sm_121.json | notes: Registry lacks an Inkling-specific artifact; hardware_scope uses closest controlled Blackwell code sm120 for sm_121.
- L1-afb4f37ca5 2026-08-08 PR #33903: [Inkling] silu_and_mul: replace helion kernels with plain Triton (#33903) | paths: python/sglang/kernels/ops/moe/inkling_moe.py; python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/silu_and_mul_interleaved_sm_100.json; python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/silu_and_mul_interleaved_sm_121.json; python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/silu_and_mul_interleaved_sm_90.json | notes: NEW artifact because Inkling MoE activation is not represented in L1 registry.

## NEW:expert_pack_mxfp4  (2 events)
- L1-2d8484740d 2026-08-26 PR #35314: Support deepseek v4 and kimi k3 on ssd (#35314) | paths: python/sglang/kernels/jit/csrc/moe/expert_pack_mxfp4.cu; python/sglang/kernels/ops/moe/expert_pack_mxfp4.py; python/sglang/srt/layers/moe/expert_pack.py; python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py | notes: 
- L1-dc2157dcd6 2026-09-10 PR #38830: [JIT] Port the expert-pack MXFP4 kernels to load_jit and fix their launch limits (#38830) | paths: python/sglang/kernels/jit/csrc/moe/expert_pack_mxfp4.cu; python/sglang/kernels/jit/csrc/moe/expert_pack_mxfp4.cuh; python/sglang/kernels/ops/moe/expert_pack_mxfp4.py | notes: New artifact not in registry.

## NEW:dsv3_router_gemm  (1 events)
- L1-7248272ccc 2025-06-29 PR #7627: Add dsv3 router gemm kernel (#7627) | paths: sgl-kernel/CMakeLists.txt; sgl-kernel/csrc/common_extension.cc; sgl-kernel/include/sgl_kernel_ops.h; sgl-kernel/python/sgl_kernel/__init__.py | notes: NEW artifact: DSV3 router GEMM is not in L1 registry but PR is router/top-k adjacent.

## NEW:modelopt_fp4_moe_adapter  (1 events)
- L1-58c468f404 2025-07-25 PR #8333: Fix FP4 MoE accuracy from missing routed_scaling_factor (#8333) | paths: python/sglang/srt/layers/quantization/modelopt_quant.py; python/sglang/srt/server_args.py | notes: No registry artifact exactly covers ModelOpt FP4 MoE adapter in this batch.

## NEW:nvfp4_expert_quant  (1 events)
- L1-4fc09e0df0 2025-08-15 PR #8777: Fp4 MOE quant kernel optimization (#8777) | paths:  | notes: Uses NEW artifact because registry has no nvfp4_expert_quant artifact.

## NEW:w8a8_fp8_quant_moe_adapter  (1 events)
- L1-2f8ba6fe82 2025-09-14 PR #10429: [Fix] MoE: fix w8a8_fp8 MoE and add tests to cover this code path (#10429) | paths: python/sglang/srt/layers/quantization/w8a8_fp8.py | notes: NEW artifact is w8a8_fp8 quantization adapter; registry has no specific L1 artifact for this quantized MoE adapter.

## NEW:ktransformers_ep_wrapper  (1 events)
- L1-ddd1440d0f 2025-11-09 PR #12834: Refactor KTransformers heterogeneous compute with unified GPU-quantization backend (#12834) | paths: python/sglang/srt/layers/moe/fused_moe_triton/layer.py; python/sglang/srt/layers/moe/kt_ep_wrapper.py | notes: NEW artifact kept for KTransformers EP wrapper adapter.

## NEW:rocm_int4fp8_moe_method  (1 events)
- L1-5af84c8af5 2026-01-14 PR #7392: [AMD][Quantization] Add `int4fp8_moe` online quantization on ROCm (#7392) | paths:  | notes: NEW artifact retained from stage-1 because no registry artifact specifically covers ROCm int4-fp8 online MoE quantizatio

## NEW:mxfp8_cutlass_moe  (1 events)
- L1-3c9cc44ff5 2026-01-29 PR #17449: Add mxfp8 support for online quantization, Triton dense linear, and CUTLASS MoE (#17449) | paths: python/sglang/srt/layers/moe/cutlass_moe.py | notes: NEW artifact retained for MXFP8 CUTLASS MoE because registry has cutlass adapters but no specific MXFP8 artifact.

## NEW:compressed_tensors_moe_scheme  (1 events)
- L1-107958a489 2026-02-09 PR #17828: Make compressed-tensors MoEs support ignored layers (#17828) | paths:  | notes: NEW artifact retained because registry has no compressed-tensors MoE scheme artifact.

## NEW:wna16_moe_method  (1 events)
- L1-7a607c4900 2026-02-16 PR #18459: fix_get_quant_method_in_fused_moe_condition (#18459) | paths: python/sglang/srt/layers/quantization/moe_wna16.py | notes: NEW artifact retained because registry lacks WNA16 MoE method artifact.

## NEW:unquant_moe_method  (1 events)
- L1-cdc411160b 2026-02-25 PR #19287: [NPU] Fix a corner case where FusedMoE.top_k is not explicitly declared (#19287) | paths: python/sglang/srt/layers/quantization/unquant.py | notes: Stage 1 NEW artifact retained for unquantized MoE quantization method wrapper.

## NEW:xpu_fused_experts_moe  (1 events)
- L1-1e9eecfa36 2026-04-13 PR #22417: [Intel GPU] Enable sgl-kernel-xpu fused_experts MoE kernel path for GPT-OSS bf16 models. (#22417) | paths: python/sglang/srt/layers/quantization/unquant.py | notes: NEW artifact retained from stage 1; no registry ID exists for XPU fused_experts MoE.

## NEW:lora_moe_align  (1 events)
- L1-917d2aa1dc 2026-04-22 PR #23178: [LoRA] Fix EP + per-expert MoE LoRA illegal memory access (#23178) | paths: python/sglang/jit_kernel/csrc/lora/moe_lora_align_kernel.cu | notes: NEW artifact retained from stage 1; LoRA-MoE align is within L1 but absent from registry.

## NEW:xpu_native_moe  (1 events)
- L1-0ac23cffac 2026-04-29 PR #12771: Add intel_xpu as backend for GptOssForCausalLM, enabled for bf16 models with torch native backend (# | paths: python/sglang/srt/layers/moe/fused_moe_native.py | notes: NEW artifact is Intel XPU torch-native MoE path.

## NEW:mxfp8_jit_moe  (1 events)
- L1-3f7c95d6cc 2026-04-29 PR #23833: [JIT Kernel][1/2]Migrate MXFP8 Group GEMM & Quant into JIT (#23833) | paths: python/sglang/jit_kernel/csrc/moe/expert_specialization/es_sm100_mxfp8_blockscaled_group_quant.cuh; python/sglang/jit_kernel/csrc/moe/expert_specialization/es_sm100_mxfp8_blockscaled_moe_group_gemm.cuh; python/sglang/jit_kernel/csrc/moe/expert_specialization/es_sm100_mxfp8_blockscaled_moe_group_gemm_functor.cuh; python/sglang/jit_kernel/csrc/moe/expert_specialization/es_sm100_mxfp8_blockscaled_moe_group_gemm_traits.cuh | notes: No registry artifact exists yet for MXFP8 JIT MoE.

## NEW:sm120_mxfp4_moe_triton  (1 events)
- L1-524ba10eda 2026-06-01 PR #24692: feat: SM120 (Blackwell Desktop) support for DeepSeek-V4 inference (#24692) | paths: python/sglang/srt/layers/moe/fused_moe_triton/mxfp4_moe_sm120_triton.py; python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py | notes: NEW artifact named from stage-1 hint.

## NEW:gemma4_fused_router  (1 events)
- L1-5ae8d286d2 2026-06-02 PR #26502: perf(gemma4): single-launch fused router (topk + softmax + scale) (#26502) | paths:  | notes: NEW artifact named from stage-1 hint.

## NEW:quark_mxfp4_moe_quant  (1 events)
- L1-293816ab14 2026-06-03 PR #18005: [AMD][MXFP4] Online MXFP4 quantization 1/N - dense and MOE models w. original BF16 weight (#18005) | paths: python/sglang/srt/layers/quantization/__init__.py; python/sglang/srt/layers/quantization/quark/quark.py; python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py; python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py | notes: NEW artifact named from stage-1 hint.

## NEW:experimental_lora_trtllm_moe  (1 events)
- L1-c9f582a272 2026-06-05 PR #27329: [LoRA] Experimental fast LoRA path with `experimental_sgl_trtllm` MoE backend for FP8 and NVFP4 mode | paths: python/sglang/jit_kernel/csrc/trtllm_lora_temp/kimi_k2_moe_fused_gate.cuh; python/sglang/jit_kernel/csrc/trtllm_lora_temp/moe_lora_merged_align_kernel.cu; python/sglang/jit_kernel/csrc/trtllm_lora_temp/topk_softmax_pack.cuh; python/sglang/jit_kernel/trtllm_lora_temp/data/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu | notes: NEW artifact named from stage-1 hint; TRT-LLM target is named in PR text, not a specific PR.

## NEW:lplb_dispatch_kernels  (1 events)
- L1-92b42c8d8a 2026-06-16 PR #24515: LPLB: linear-programming load balancer for MoE expert parallelism (#24515) | paths: python/sglang/srt/layers/moe/hash_topk.py; python/sglang/srt/layers/moe/topk.py; python/pyproject.toml; python/sglang/srt/eplb/expert_location_dispatch.py | notes: NEW artifact is the LPLB JIT dispatch/solver kernel family.

## NEW:inkling_gate_topk  (1 events)
- L1-02236fa38c 2026-07-19 PR #31681: Add Inkling model support (#31681) | paths: python/sglang/jit_kernel/csrc/moe/inkling_gate_topk_renorm.cuh; python/sglang/jit_kernel/csrc/trtllm_lora_temp/moe_lora_merged_align_kernel.cu; python/sglang/jit_kernel/trtllm_lora_temp/data/csrc/trtllm_fused_moe_kernel_launcher.cu; python/sglang/jit_kernel/trtllm_lora_temp/data/csrc/trtllm_fused_moe_runner.cu | notes: NEW:inkling_gate_topk is not in registry; PR introduces model-specific gate/topk kernel.

## NEW:dwdp_moe_prefill  (1 events)
- L1-37a830b667 2026-07-20 PR #29778: [Feature] Add DWDP (Distributed Weight Data Parallelism) for MoE prefill (#29778) | paths: python/sglang/srt/layers/moe/dwdp/__init__.py; python/sglang/srt/layers/moe/dwdp/dwdp_manager.py; python/sglang/srt/layers/moe/dwdp/layout.py; python/sglang/srt/layers/moe/dwdp/page_pool.py | notes: NEW artifact is the DWDP prefill manager/transport/weight-buffer integration.

## NEW:longcat_router_hpc_gemm  (1 events)
- L1-e4eea7ce2f 2026-07-21 PR #30247: Optimize LongCat-Flash router GEMM with the HPC-Ops bf16xfp32 kernel (#30247) | paths:  | notes: NEW artifact is LongCat router GEMM integration around upstream HPC-Ops.

## NEW:aot_router_gemm  (1 events)
- L1-03342e7732 2026-07-22 PR #30280: Delete sgl-kernel AOT router GEMM and fused A GEMM (#30280) | paths: sgl-kernel/CMakeLists.txt; sgl-kernel/csrc/common_extension.cc; sgl-kernel/csrc/common_extension_musa.cc; sgl-kernel/include/sgl_kernel_ops.h | notes: Registry has no AOT router GEMM artifact; used NEW for deleted AOT CUDA kernels.

## NEW:hpc_ops_moe_runner  (1 events)
- L1-3d91a569ce 2026-07-24 PR #30541: [MoE Backend] Add HPC-Ops FP8 MoE runner backend (#30541) | paths: python/sglang/srt/layers/moe/fused_moe_triton/layer.py; python/sglang/srt/layers/moe/moe_runner/hpc_ops.py; python/sglang/srt/layers/moe/moe_runner/runner.py; python/sglang/srt/layers/moe/token_dispatcher/standard.py | notes: NEW artifact is optional HPC-Ops runner adapter.

## NEW:kimi_k3_radix4_topk  (1 events)
- L1-edd675cecf 2026-08-22 PR #34490: [AMD] Add Radix-4 MoE top-k router kernel for Kimi-K3 routing (#34490) | paths: python/sglang/kernels/jit/csrc/moe/route_radix4_hip.cuh; python/sglang/kernels/ops/moe/moe_route_radix4.py; python/sglang/srt/layers/moe/topk.py | notes: 

## NEW:shuffle_rows_with_scales  (1 events)
- L1-77940dec80 2026-08-24 PR #34915: [MoE] Gather the cutlass MoE activation and its scales in one launch (#34915) | paths: python/sglang/kernels/ops/moe/__init__.py; python/sglang/kernels/ops/moe/shuffle_rows_with_scales.py; python/sglang/srt/layers/moe/cutlass_moe.py | notes: 

## NEW:mxfp4_moe_quant  (1 events)
- L1-2935bb8e79 2026-08-26 PR #36456: Fix OOB read in mxfp4 MoE weight scales on Hopper (#36456) | paths:  | notes: 

## NEW:pack_topk_ids  (1 events)
- L1-e57e934bcc 2026-09-01 PR #32882: [Bugfix] Accept int64 top-k IDs in FlashInfer routed MoE packer (#32882) | paths: python/sglang/kernels/ops/moe/pack_topk_ids.py | notes: 

## NEW:native_moe  (1 events)
- L1-acea43079f 2026-09-02 PR #36407: Fix native MoE handling of noncontiguous top-k IDs (#36407) | paths: python/sglang/srt/layers/moe/fused_moe_native.py | notes: 

## NEW:router_gemv  (1 events)
- L1-5177a3ec08 2026-09-08 PR #36557: MiniMax-M3: Triton split-K router GEMV with in-kernel fixup (#36557) | paths: python/sglang/srt/models/minimax_m3.py | notes: New artifact not in registry.

## NEW:glm_nextn_moe_quant  (1 events)
- L1-f920be4b09 2026-09-15 PR #39155: [AMD] GLM-5.2 NextN: cast draft fused MoE to per-channel FP8 (#39155) | paths: python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8_moe.py; python/sglang/srt/models/glm4_moe.py | notes: New quantization-policy artifact not in registry.

## NEW:smallm_moe_gfx950  (1 events)
- L1-32290dda2c 2026-09-23 PR #40204: [AMD] Small-M MXFP4 fused-MoE kernel for gfx950 (Qwen) (#40204) | paths: python/sglang/kernels/ops/moe/smallm_moe_gfx950/__init__.py; python/sglang/kernels/ops/moe/smallm_moe_gfx950/smallm_moe.hip; python/sglang/srt/layers/moe/moe_runner/aiter.py | notes: NEW artifact is the gfx950 small-M HIP kernel package under kernels.ops.moe.smallm_moe_gfx950.

## NEW:moe_sorting_small  (1 events)
- L1-36f59982fa 2026-09-24 PR #36559: MoE: small-batch sorting path with fused mxfp8 quantisation (#36559) | paths: python/sglang/kernels/ops/moe/moe_sorting_small.py; python/sglang/kernels/ops/moe/mxfp8_moe_amd_gfx95.py; python/sglang/srt/layers/moe/moe_runner/aiter.py; python/sglang/srt/layers/moe/moe_runner/triton.py | notes: NEW artifact is kernels.ops.moe.moe_sorting_small and its runner hooks.

## NEW:rocm_router_gate  (1 events)
- L1-effb752188 2026-09-26 PR #41019: dsv4.1-amd: KV cache layouts, FP4 indexer, compressor and router kernels (#41019) | paths: python/sglang/kernels/ops/moe/rocm_router_gate.py | notes: NEW artifact is the ROCm Triton router_gate operation.
