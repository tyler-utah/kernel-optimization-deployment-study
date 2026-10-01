# L2 family tree
- 2024-07-26 #693 DeepSeek-V2 model file.
  - 2024-08-05 #905 MLA forward, memory, Triton adaptation.
    - 2024-08-19 #1138 Triton MLA/GQA/MQA decode; comments cite LightLLM.
    - 2024-08-30 #1261 offline transposed absorbed weights.
    - 2024-12-05 #2349 prefill without weight absorption.
    - 2026-02-27 #19122 split forward MLA into `deepseek_common/.../forward_mla.py`.
  - 2024-09-17 #1447 MLA default; 2024-09-10 #1380 named attention backend selection.
- FlashInfer: #3550 initial support; #3785 dedicated `flashinfer_mla_backend.py`; #3967 ragged prefill; #4012 fast plan.
- FlashMLA: #4472 backend; #4514 CUDA graph; #11717 sgl-kernel build/wrapper; #32648 moved to `kernels/aot`.
- CUTLASS: #5142 SM100 kernel; #5390 backend; #6929 SM100 templates; #32114 deleted/replaced by fallbacks.
- FA3/FA4: #4831 FA3 MLA support; #5210 FA3 default on Hopper.
- TRT-LLM: #8632 `trtllm_mla_backend.py`; later FP8/spec/DCP adaptations.
- ROCm: #3237 `rocm_mla_decode_rope.py`; AITER MLA path; #24933 `hip_flash_mla.py`; #34647 `aiter_mla_gluon.py`.
- Later branches: #24925 tokenspeed_mla; #24737/#32612 CuTe-DSL; #24692 SM120; NPU #7722/#13359; sparse adapters #21783/#29775.

## Ten important ancestry facts
1. Early Triton decode explicitly cites LightLLM token-attention kernels in source comments.
2. #905 is the first durable MLA implementation and says it adds MLA forward plus memory/Triton adaptation.
3. Weight absorption is model-side algebra (`w_kc/w_vc`), later moved to `forward_mla.py`.
4. Dispatch evolved from boolean MLA flags to named `--attention-backend` selection.
5. #3785 split FlashInfer MLA from the general FlashInfer backend.
6. FlashMLA backend subclasses FlashInfer MLA but wraps `flash_mla_with_kvcache`/`get_mla_metadata`.
7. FlashMLA delivery moved from external package integration to in-tree sgl-kernel/libtorch build (#11717), then `kernels/aot` (#32648).
8. CUTLASS MLA was removed before cutoff; #32114 says SM10.0-only, disabled on GB300, no CI, with FlashInfer fallback.
9. TRT-LLM MLA is via FlashInfer; the file docstring states "TRTLLM MLA kernels from flashinfer".
10. `MLATokenToKVPool`, `set_mla_kv_buffer`, `concat_mla`, and FP8 pack/quantize kernels encode the cache layout contract.

## Open questions
- Exact upstream FlashInfer SHAs for MLA wrapper/TRTLLM/CuTe kernels.
- Exact first AITER MLA change in `aiter_backend.py`.
- Separate FA4 upstream ancestry.

## Default-backend timeline
| hardware/model scope | timeline |
|---|---|
| sm80 / other CUDA DeepSeek MLA | #905 optional `--enable-mla`; #1447 MLA default on Triton; #5210+ remains Triton fallback when not Hopper/Blackwell; cutoff generic sm80/sm120 fallback is Triton unless model-specific override applies. |
| sm90 / Hopper DeepSeek MLA | #1447 Triton default; #5210 switches default to FA3 on Hopper CUDA>=12.3; cutoff generic sm90 remains FA3. |
| sm100 / Blackwell DeepSeek MLA | #5210 era generic Blackwell selected FlashInfer; #8632 code shows sm100 default FlashInfer; cutoff DeepSeek-V3 override selects `trtllm_mla` when unset, while DeepSeek-V4 uses `dsv4`/FlashMLA sparse path and `trtllm` only opt-in FP8. |
| sm120 | introduced later as SM120 Flash MLA/sparse support (#24692/#27455); cutoff generic MLA falls back to Triton, with DeepSeek-V4 dsv4 path and FlashInfer sparse/SM120 helpers for sparse branches. |
| ROCm | #3237 adds ROCm Triton MLA decode+RoPE; by #8632 auto default is AITER only for KV heads 16/128 and no speculative decoding, otherwise Triton; cutoff keeps ROCm-specific MLA forward handlers. |
| NPU / other | Ascend PA/MLA arrives #7722; by #8632 NPU auto default is Ascend; #13359 adds NPU MLA modules. CPU/MPS use torch_native/intel_amx where supported. |

