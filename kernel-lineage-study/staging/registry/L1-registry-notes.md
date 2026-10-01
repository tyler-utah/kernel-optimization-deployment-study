# L1 registry notes

## Family tree
- vLLM ancestry: align 2024-01-29 #2453 -> align+sum 2024-10-24 #9384 -> libtorch_stable 2026-06-10 #44565; topk_softmax 2024-02-05 #2769 -> SGLang port #4302.
- SGLang Triton fused_moe: layers/fused_moe.py 2024-05-27 #484 -> package #1095 -> fused_moe_triton/Grok #2163 -> layers/moe #2563 -> triton_utils #23019; helper kernels moved to kernels.ops #30786.
- Align: AOT CUDA #2579 -> moves #4025/#4213/#32648; Triton align exact intro 2025-01-02 #2712; JIT copy 2026-04-08 -> kernels/jit #32072; single-token #32541; small-numel #32395; .hip variant added/reverted 2025-03-07 #4178/#4186.
- Routing: topk.py #2563; fused_gate/grouped top-k #4530 -> JIT/ops; topk_sigmoid #13049; HashTopK #23882; router.py #4398 -> kernels.ops/router.py #30786.
- Reduction/scaling: Triton moe_sum_reduce split #9878; CUDA moe_sum_reduce #10321 and moe_sum #11019; JIT topk_sum/topk_reduce #32890. routed_scaling_factor fusion is in fused_gate/topk output scaling and in moe_sum_reduce combine scaling.
- EP: EPMoE layer 2024-12-06 -> layers/moe #2563; ep_moe kernels moved #30786; DeepEP dispatcher 2025-03-19 -> token_dispatcher/deepep #8658; other a2a dispatchers Mooncake #10423, FuseEP #12078, FlashInfer #14668, MorIEP #17012, NIXL #19248.
- Runners: MoeRunner #9269; Triton runner #9269; OpenAI triton_kernels #7689/#11795; DeepGEMM #11211; Marlin #14554; AITER #23597; DeepGEMM MegaMoE #23882.
- FlashInfer split: trtllm #15151/#18184; CuteDSL #9199 then runner #21339 (same adapter lineage); MXFP4 #26489 deleted/merged #28211; CUTLASS centralized #28211; MegaMOE #31470.
- CUTLASS split: FP8 blockwise #5281; Python adapters #5694/#7762; NVFP4 #6093 ended with JIT deletion/replacement; W4A8 #7762 plus AOT move #32648.
- Misc: fused_moe_patch 2024-09-24..2024-12-24; Grok fused_moe 2024-11-24..27; CPU/NPU/MUSA/Ascend are hardware-scoped artifacts.

## Ten ancestry facts
1. Initial SGLang fused_moe header says “Adapted from” vLLM fused_moe.py blob c7f2cf2.
2. Initial SGLang fused_moe called `ops.moe_align_block_size`, tying pre-sgl-kernel alignment to vLLM custom ops.
3. vLLM #9384 moved align into moe_align_sum and says it replaced `torch.sum` with customized `moe_sum`.
4. SGLang #4302 explicitly “Cherry picked current vllm MoE topk softmax kernel template”.
5. vLLM later ported SGLang align: #12574 cites SGLang ded9fcd/ba5112ff; #19572 says implementation taken from SGLang 8b5f83ed.
6. Fused MoE file identity survives moves #1095/#2163/#2563/#23019.
7. #23019 says TritonRunnerCore.run and fused_experts_impl had ~95% identical logic and reconciled both.
8. #30786 says fused_moe_triton_kernels and ep_moe kernels were byte-identical moves to sglang.kernels.ops.moe.
9. FlashInfer CuteDSL layer and MoeRunner file are one adapter lineage: 2025 wrapper use, 2026 standard `--moe-runner-backend flashinfer_cutedsl` integration.
10. attention/dsv4 topk_v2 is excluded; MoE HashTopK instead calls kernels.ops.moe.dsv4.hash_topk.

## Open questions
- Some hardware/small-wrapper PR intent remains census-only.
- Not every rename was individually re-diffed with `git diff -M`.
- Grok fused_moe deletion successor is unclear.
- OpenAI triton_kernels and AITER pins were not found; adapters appear optional-import based.
