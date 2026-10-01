### L3-472d8c9125  (L3, 2026-09-08, sha 472d8c912566, PR #55629)
TITLE: [Perf] Fuse DeepEncoder relative bias in Triton attention (#55629)
SOURCES: subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/deepencoder.py (+191/-7)
LABELS: performance, ready, multi-modality
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ DeepEncoder relative attention currently expands decomposed height and width ⏎ terms into a dense `[B, heads, HW, HW]` bias before SDPA. Its 64x64 global ⏎ attention therefore materializes a 384 MiB BF16 temporary. ⏎  ⏎ This PR adds a dedicated Triton forward kernel that evaluates both relative ⏎ terms while streaming key/value tiles through online softmax: ⏎  ⏎ ```text ⏎ before: rel_h + rel_w -> dense 4D bias -> SDPA ⏎ after:  compact rel_h/rel_w -> f …[truncated]

### L3-414057a3d3  (L3, 2026-09-08, sha 414057a3d3aa, PR #52156)
TITLE: [Bugfix] Apply attention sinks in the Transformers backend (#52156)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/transformers/test_backend.py (+55/-11); vllm/model_executor/models/transformers/__init__.py (+25/-0); vllm/model_executor/models/transformers/base.py (+14/-18); vllm/model_executor/models/transformers/fusers/attention.py (+33/-2)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ vLLM PR #48270 adds native `GraniteSWAForCausalLM` / `GraniteMoeSWAForCausalLM` because "the transformers backend does NOT handle these models correctly. Sink tokens get dropped silently and so the model gives wrong output." That is accurate, and it is a bug in the Transformers modeling backend rather than something inherent to those models, so this PR fixes it there and every current and future sink model works on the backend. ⏎  ⏎ T …[truncated]

### L3-db3814a4f2  (L3, 2026-09-08, sha db3814a4f215, PR #55780)
TITLE: [Attention] Require explicit DCP support from attention implementations (#55780)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.flashmla_v1_adapter, L3.mla.cutlass_v1_backend, L3.mla.flashattn, L3.mla.flashinfer, L3.mla.flashmla_sparse, L3.mla.flashinfer_sparse, L3.mla.rocm_aiter, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backend.py (+1/-1); vllm/v1/attention/backends/flash_attn.py (+1/-0); vllm/v1/attention/backends/flashinfer.py (+1/-0); vllm/v1/attention/backends/mla/cutlass_mla.py (+1/-0); vllm/v1/attention/backends/mla/flashattn_mla.py (+1/-0); vllm/v1/attention/backends/mla/flashinfer_mla.py (+1/-0); vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py (+1/-0); vllm/v1/attention/backends/mla/flashmla.py (+1/-0); vllm/v1/attention/backends/mla/flashmla_sparse.py (+1/-0); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+1/-0); (+5 more)
LABELS: rocm, ready, nvidia
BODY: - Default `AttentionImplBase.supports_dcp` to `False` so implementations must explicitly opt into DCP. ⏎ - Enable eleven existing DCP implementation classes, preserving thirteen registered backends, including FlashAttention, FlashInfer, and the supported MLA variants. ⏎ - Reject unsupported ROCm, standard Triton, FlexAttention, and TurboQuant DCP configurations during selection, before loading model weights. ⏎ - Preserve the existing sparse AITER ML …[truncated]

### L3-bfb443a6b6  (L3, 2026-09-08, sha bfb443a6b6f6, PR #55924)
TITLE: [Kimi Bug] Fix kda ima `Triton Error [CUDA]: an illegal memory access was encountered` (#55924)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/models/kimi_k3/nvidia/kda.py (+1/-1)
LABELS: bug, ready, nvidia, kimi, k3
BODY: ## Purpose ⏎  ⏎ ```bash ⏎  ⏎ vllm serve moonshotai/Kimi-K3 \ ⏎   --trust-remote-code \ ⏎   --tensor-parallel-size 8 \ ⏎   --load-format fastsafetensors \ ⏎   --gpu-memory-utilization 0.9 \ ⏎   --enable-prefix-caching \ ⏎   --reasoning-parser kimi_k3 \ ⏎   --enable-auto-tool-choice \ ⏎   --tool-call-parser kimi_k3 \ ⏎   --host 0.0.0.0 \ ⏎   --port 30000 \ ⏎   --max-model-len auto \ ⏎   --max-num-seqs 48 \ ⏎   --speculative-config '{"model":"RedHatAI/Kimi-K3-specul …[truncated]

### L3-755541443f  (L3, 2026-09-08, sha 755541443f62, PR #53856)
TITLE: [Bugfix][ROCm] Mask paged attention V cache padding (#53856)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.rocm.custom_paged
FILES: csrc/rocm/attention.cu (+196/-31); tests/kernels/attention/test_attention.py (+26/-10)
LABELS: bug, rocm, ready
BODY: ## Purpose ⏎  ⏎ ROCm paged attention masks logits for tokens beyond `seq_len`, but its vectorized V-cache loads can still consume unwritten slots in the final block. If one of those slots contains NaN, the subsequent `0 * NaN` accumulation propagates NaNs into the output. K does not need the same treatment because invalid K values are removed before softmax; V must be sanitized before the matrix accumulation. ⏎  ⏎ This PR keeps the vectorized loads and h …[truncated]

### L3-13cf9e05c1  (L3, 2026-09-08, sha 13cf9e05c1ed, PR #55170)
TITLE: [Perf][Quant][NVFP4] Prefer W4A4 linear kernels over weight-only ones on SM120/121 (#55170)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/quantization/test_nvfp4_kernel_selection.py (+32/-0); vllm/model_executor/kernels/linear/__init__.py (+2/-1)
LABELS: quantization
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ On SM120/121, NVFP4 linear layers run a **weight-only (W4A16)** kernel even when the checkpoint can feed FP4 activations, so the FP4 tensor cores sit idle. ⏎  ⏎ Measured on Qwen3.8-27B-NVFP4, RTX PRO 6000 Blackwell Max-Q, BS1, 32K cached prefix + 2K ISL + 256 OSL, same node, same checkpoint, `v0.28.1rc1.dev337+g27a94d1ce` with the V2 model runner: ⏎  ⏎ | `--linear-backend` | selected kernel | TTFT | output tok/s | ⏎ |---|---|---|---| ⏎ | `auto` ( …[truncated]

### L3-f998862d46  (L3, 2026-09-08, sha f998862d46e7, PR #53379)
TITLE: [Bugfix] Fix Kimi K3 loading with interleaved weight streams (#53379)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/kimi_k3/test_weight_loading.py (+72/-0); vllm/models/kimi_k3/nvidia/model.py (+8/-2)
LABELS: bug, ready, kimi, k3
BODY: ## Purpose ⏎  ⏎ Fix Kimi K3 weight loading when a streaming loader yields non-contiguous top-level module prefixes. ⏎  ⏎ `AutoWeightsLoader` groups adjacent prefixes. A stream such as ⏎ `language_model.*`, `vision_tower.*`, `language_model.*` therefore invokes the ⏎ Kimi language-model loader more than once. Previously every invocation ⏎ finalized MegaMoE weights and removed the original weight and scale parameters, ⏎ so a later language-model group could fail w …[truncated]

### L3-d29c88f162  (L3, 2026-09-08, sha d29c88f162a3, PR #53007)
TITLE: [Core] Let SWA layers take the primary block size to avoid inflating the KV block LCM (#53007)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention/attention.py (+20/-10); tests/v1/attention/test_backend_per_kind.py (+22/-1); vllm/v1/core/kv_cache_utils.py (+3/-0)
LABELS: ready, kv-cache-manager
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ An SWA layer started from the backend's smallest kernel block, so its ⏎ page divided a larger MLA page and unify scaled it by an integer ratio ⏎ to a block size coprime with the primary one (1024 B/token SWA draft ⏎ next to a 1152 B/token MLA target: 1728 vs 1536, scheduler LCM 13824), ⏎ which destroys prefix-cache granularity. ⏎  ⏎ When the backend can run the primary block size unsplit (e.g. FlashAttn ⏎ via backend_per_kind.sliding_windo …[truncated]

### L3-9c2d21046b  (L3, 2026-09-08, sha 9c2d21046bb3, PR #54788)
TITLE: [Bugfix][Spec Decode] Honour the draft's moe_backend on Model Runner V2 (#54788)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/spec_decode/test_draft_moe_backend_override.py (+87/-0); vllm/v1/worker/gpu/spec_decode/eagle/utils.py (+10/-0)
LABELS: bug, speculative-decoding, ready, mrv2
BODY: ## Purpose ⏎  ⏎ The draft model is loaded with the target's `VllmConfig`, so `--speculative-config '{"moe_backend": ...}'` is ignored on Model Runner V2. When the target is quantized and the draft is not, which is the normal shape of an MTP head, a quantized-only backend rejects the draft and the engine fails to start: ⏎  ⏎ ``` ⏎ ValueError: moe_backend='flashinfer_b12x' is not supported for unquantized MoE. ⏎ Expected one of ['triton', 'batched_triton', 'fl …[truncated]

### L3-c268198715  (L3, 2026-09-08, sha c268198715de, PR #55472)
TITLE: [Bugfix][Spec Decode] Preserve target parallel config (DCP) for DSpark (#55472)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/spec_decode/dspark/utils.py (+7/-1)
LABELS: bug, speculative-decoding, ready, mrv2, dflash
BODY: ## Purpose ⏎  ⏎ Fix Kimi-K3 startup with DSpark when decode context parallelism (DCP) is enabled. ⏎  ⏎ #50514 changed `load_dspark_model` to use `speculative_config.draft_parallel_config` instead of target's `parallel_config` when constructing the draft `VllmConfig`. ⏎ However, `speculative_config.draft_parallel_config` only copies a subset of the target parallel settings and leaves DCP fields at their defaults (dcp=1) (in function `create_draft_paral …[truncated]

### L3-bcca76e7db  (L3, 2026-09-08, sha bcca76e7dbca, PR #55908)
TITLE: [Test] Dequantize NVFP4 KV cache scales in the layout the kernel writes (#55908)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/kernels.yaml (+4/-0); tests/kernels/attention/test_cache.py (+17/-4); tests/kernels/attention/test_flashinfer_trtllm_attention.py (+1/-1); tests/kernels/quantization/nvfp4_utils.py (+20/-12)
LABELS: ready, ci/build, nvidia, quantization
BODY: ## Purpose ⏎  ⏎ `test_reshape_and_cache_flash[nvfp4]` fails on every Blackwell part with the current kernel, and nobody sees it because no CI job runs it on Blackwell. ⏎  ⏎ `reshape_and_cache_flash` writes K block scales linearly and V block scales in the 4x4 swizzle of the SM100 trtllm-gen reader (`nvfp4_kv_cache_kernels.cu`, `kv == 0` branch vs `swizzle_scale_offset`). The test reference `dequant_nvfp4_kv_cache` unswizzles both, so the K comparison run …[truncated]

### L3-9e137d8d23  (L3, 2026-09-08, sha 9e137d8d230c, PR #51925)
TITLE: [Kernel] Enable optimized FlashInfer add-RMSNorm NVFP4 fusion (#51925)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/compile/passes/test_fusion.py (+103/-0); vllm/compilation/passes/fusion/rms_quant_fusion.py (+169/-0); vllm/compilation/passes/utility/fix_functionalization.py (+14/-0)
LABELS: ready, torch.compile
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: This PR follows up on #36413, which explored the same fusion. Since then, the FlashInfer fused add-RMSNorm + NVFP4 kernel received performance improvements in [flashinfer#4494](https://github.com/flashinfer-ai/flashinfer/pull/4494). With these improvements, the kernel brings up positive performance influence on some workloads. ⏎  ⏎   For `nvidia/Qwen3-30B-A3B-NVFP4`, the existing path still uses two separate kernels: ⏎  ⏎   1. An Inductor-generated T …[truncated]

### L3-1b2c591cd0  (L3, 2026-09-08, sha 1b2c591cd0c3, PR #55031)
TITLE: [Bugfix] speedup nvfp4 kv for FMHA (#55031)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+12/-4); tests/compile/passes/test_fusion_attn.py (+55/-19)
LABELS: bug, ready, torch.compile, nvidia
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: Do not do type conversion inside copy_. ⏎  ⏎ Guard NVFP4 output fusion when the KV cache is NVFP4. ⏎  ⏎ Usage: ⏎ ``` ⏎   vllm serve <model> --kv-cache-dtype nvfp4 \ ⏎     --compilation-config '{"pass_config": {"fuse_attn_quant": true}}' ⏎ ``` ⏎  ⏎ AttnQuantFusionPass fuses the attention output quant so the trtllm-gen FMHA ⏎ writes FP8/NVFP4 straight into o_proj's input, skipping the FP8 -> BF16 -> ⏎ quant round trip. With an FP8 o_proj this works out of the  …[truncated]

### L3-4aadb0f14d  (L3, 2026-09-08, sha 4aadb0f14d4c, PR #53491)
TITLE: [Core] Enhance cpu<->gpu sync checking to include paged async copies (#53491)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/utils_/test_gpu_sync_debug.py (+146/-0); vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py (+12/-4); vllm/envs.py (+5/-3); vllm/model_executor/offloader/uva.py (+10/-7); vllm/utils/gpu_sync_debug.py (+159/-0)
LABELS: rocm, ready, qwen, kv-connector, nvidia, glm
BODY: The CPU<->GPU sync checking added in https://github.com/vllm-project/vllm/pull/44800 / https://github.com/vllm-project/vllm/pull/43107 makes use of pytorch's `torch.cuda.set_sync_debug_mode` function. ⏎  ⏎ However this does not catch stalls due to "non-blocking" tensor copies to or from non-pinned host memory, where an implicit copy is made to a staging buffer. This also includes copies from pinned tensors which have non-dense layouts. ⏎  ⏎ This PR e …[truncated]

### L3-bc8587f829  (L3, 2026-09-08, sha bc8587f82944, PR #55099)
TITLE: [ROCm][Perf][Bugfix] Multi-stream perf improvements; rocprofiler fixes (#55099)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.rock_base (+80/-0); docker/Dockerfile.rocm_base (+81/-85); tests/kernels/moe/parallel_utils.py (+0/-12); tests/kernels/moe/test_deepep_moe.py (+0/-7)
LABELS: bug, rocm, ready, ci/build
ISSUES: #51644 [CI Failure]: tests/kernels/moe/test_deepep_moe.py SIGSEGV on ROCm, drop the two teardown workarounds once the image ships rocm-systems#6942
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ This PR does the following: ⏎  ⏎ 1. Relocates the ROCR+CLR build to its separate stage, instead of being a prerequisite of the `base` stage. This allows ROCR+CLR updates to not invalidate + force rebuilds of all build stages. Downstream libraries currently only ever link against them dynamically, so this relocation is safe. ⏎  ⏎ 2. Brings a backport of changes in https://github.com/ROCm/rocm-systems/pull/11212 to ROCm 7.2.4's ROCR+CLR, wh …[truncated]

### L3-1a522b6949  (L3, 2026-09-08, sha 1a522b69491f, PR #54112)
TITLE: [ROCm] [Docker] Upgrade default AINIC repo to ship libionic 54.0-187-1 (#54112)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.rocm (+3/-2); docker/Dockerfile.rocm_gfx1250 (+3/-2); docs/features/moriio_connector_usage.md (+1/-1)
LABELS: documentation, rocm, ready, ci/build
BODY: Bump AINIC_VERSION from 1.117.3-hydra to 1.117.5 so images install libionic1=54.0-187-1 on MI355x, which is required for MoRI-EP and NIXL on Pollara/AINIC without mounting host userspace libraries.

### L3-28a73ccfba  (L3, 2026-09-08, sha 28a73ccfba9e, PR #53319)
TITLE: [Kernel] Add NVFP4 support to the torch linear backend (#53319)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/quantization/test_nvfp4.py (+2/-0); vllm/model_executor/kernels/linear/__init__.py (+6/-0); vllm/model_executor/kernels/linear/nvfp4/pytorch.py (+94/-0)
LABELS: ready
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎  ⏎ Extend the existing `torch` linear backend to ModelOpt W4A4 NVFP4 models on ⏎ SM10x (Blackwell). ⏎  ⏎ The adapter reuses vLLM's NVFP4 quantization and layout utilities, calls native ⏎ `torch._scaled_mm`, applies the ModelOpt global dequantization scale and bias, ⏎ and lets `torch.compile`/TorchInductor select an enabled scaled-GEMM backend. ⏎ Existing optimized `auto` candidates retain priority; Torch is placed ⏎ immediately before emulation and ca …[truncated]

### L3-613ab20f7f  (L3, 2026-09-08, sha 613ab20f7f90, PR #55817)
TITLE: [Model] Enable torch.compile for Sarvam MLA (#55817)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/sarvam.py (+11/-0)
LABELS: ready, torch.compile
BODY: ## Purpose ⏎  ⏎ Enable `torch.compile` for `SarvamMLAModel` by adding `support_torch_compile`. Explicit dynamic dimensions cover `input_ids`, `positions`, `intermediate_tensors`, and `inputs_embeds`, since postponed annotations prevent the decorator from inferring them. ⏎  ⏎ This applies only the compile changes from the existing Sarvam patch. `SarvamMoEForCausalLM` already inherits compile support through `BailingMoeModel`. No open PR covering Sarvam co …[truncated]

### L3-8e1f97e709  (L3, 2026-09-08, sha 8e1f97e70984, PR #55890)
TITLE: [Qwen3.8-Flash-Next] Tune FP8 TP2/TP4 Triton MoE on B200 (#55890)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/configs/E=512,N=160,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[32,32].json (+115/-0); vllm/model_executor/layers/fused_moe/configs/E=512,N=320,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[64,64].json (+115/-0)
LABELS: qwen
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Purpose ⏎  ⏎ Add missing B200 Triton MoE configurations for Qwen3.8-Flash-Next-FP8 at TP2 (local intermediate 320, FP8 blocks 64×64) and TP4 (160, 32×32), with E=512 and top-k=10. Tune 14 token counts from 1 to 8192 for each shape. These shapes differ from #43506 and are independent of the FlashInfer loader change in #55867. TP1 keeps native 128×128 blocks and defaults to TRTLLM on B200, so it is outside this Triton tuning change. ⏎  ⏎ ## Test Plan ⏎  ⏎ # …[truncated]

### L3-5af4cc33ec  (L3, 2026-09-08, sha 5af4cc33ec08, PR #55941)
TITLE: [Bugfix] Fix OpenPangu multimodal embedding merge (#55941)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/multimodal/test_openpangu_vl.py (+55/-0); vllm/model_executor/models/openpangu_vl.py (+13/-10)
LABELS: bug, ready, multi-modality
BODY: ## Purpose ⏎  ⏎ This PR fixes `OpenPanguVLForConditionalGeneration.get_input_embeddings` when multimodal embeddings are provided. The method passed the obsolete `(input_ids, inputs_embeds, multimodal_embeddings, placeholder_token_ids)` ⏎ argument shape to `SupportsMultiModal.embed_input_ids`, which now accepts `input_ids`, `multimodal_embeddings`, and a keyword-only `is_multimodal` mask. As a result, direct multimodal calls raised `TypeError`. This  …[truncated]

### L3-5acd95906b  (L3, 2026-09-09, sha 5acd95906b7c, PR #55458)
TITLE: [Bugfix] Exclude DP token padding from draft attention metadata (#55458)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py (+4/-2)
LABELS: bug, speculative-decoding, ready, mrv2
BODY: DeepSeek-V3.2 DP8 + EP + MTP crashes when DP ranks have unequal request counts. A rank with 12 requests receives model inputs padded to 13 tokens; the sparse indexer then fails with `shape '[12, -1, 64, 128]' is invalid for input of size 106496`. This is confirmed in [main build 87584](https://buildkite.com/vllm/ci/builds/87584#01a07dad-00e7-4a29-9abe-6cdc65c9100f). The subsequent GSM8K result of 0.0000 reflects the engine failure. ⏎  ⏎ Autoregressiv …[truncated]

### L3-385dce36bc  (L3, 2026-09-09, sha 385dce36bcee, PR #54640)
TITLE: [CI/Build][Hardware][NVIDIA] Add public CUDA 13.4 Rubin build path (#54640)
SOURCES: dependency_pin, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+28/-8); requirements/rubin-prerelease.txt (+16/-0); docs/getting_started/installation/gpu.cuda.inc.md (+19/-25)
LABELS: documentation, ready, ci/build, nvidia
BODY: NVIDIA now publishes a CUDA 13.4 developer-preview apt repo publicly (`packages.nvidia.com`), and `pytorch/manylinux2_28-builder` already has a `cuda13.4`-tagged image on Docker Hub. This adds an `INSTALL_RUBIN_PRERELEASE` (#53443) path that uses both, so a fully public build is possible. ⏎  ⏎ ## What's in this PR ⏎  ⏎ - `BUILD_BASE_IMAGE`: `pytorch/manylinux2_28-builder:cuda13.4` (and the aarch64 equivalent) - already bundles a full CUDA 13.4 toolkit, n …[truncated]

### L3-3116c5d06b  (L3, 2026-09-09, sha 3116c5d06bfe, PR #54371)
TITLE: [Qwen4Exp] Support UVA PLE-offload and Engram tensor parallelism (#54371)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/engine/test_arg_utils.py (+33/-0); tests/models/qwen4_exp/test_ple.py (+327/-21); tests/test_config.py (+154/-0); vllm/config/__init__.py (+3/-0); vllm/config/engram.py (+82/-0); vllm/config/vllm.py (+28/-0); vllm/distributed/device_communicators/cuda_communicator.py (+2/-1); vllm/distributed/parallel_state.py (+38/-2); vllm/engine/arg_utils.py (+6/-0); vllm/envs.py (+6/-0); (+6 more)
LABELS: new-model, speculative-decoding, ready, ci/build, qwen, kv-connector, nvidia, quantization, mrv2, scheduler
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Add UVA-based PLE offload and Engram tensor parallelism (ETP) for Qwen3.8-Flash-Next. ⏎  ⏎ `--engram-config` controls PLE weight storage and embedding sharding: ⏎  ⏎ - `cpu_offload=true` stores each local PLE weight shard in pinned CPU memory. The GPU reads the required rows directly through CUDA Unified Virtual Addressing (UVA), using a dedicated CUDA stream. Offload can also be enabled with `VLLM_PLE_CPU_OFFLOAD=1` when `cpu_offload`  …[truncated]

### L3-cc4210f671  (L3, 2026-09-09, sha cc4210f6719e, PR #55899)
TITLE: [Perf] Improve BF16x3 router GEMM accuracy and make it default on sm100 (#55899)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/test_bf16x3_router_gemm_cutedsl.py (+14/-4); vllm/config/kernel.py (+0/-3); vllm/engine/arg_utils.py (+0/-7); vllm/model_executor/layers/fused_moe/router/bf16x3_router_gemm_cutedsl.py (+249/-56); vllm/model_executor/layers/fused_moe/router/gate_linear.py (+4/-16); vllm/model_executor/warmup/kernel_warmup.py (+47/-2)
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Follow up on #47973. This only concerns router GEMM with FP32 weights i.e. BF16xFP32->FP32 ⏎  ⏎ This PR adds master rmem accumulator for long accumulation chain, restoring accuracy against FP32, hence making BF16x3 GEMM suitable as FP32 GEMM replacement whenever the kernel is supported (sm100 family). Custom op wrapper is needed for torch.compile support since torch.compile-based models will execute this new kernel as well. ⏎  ⏎ I also  …[truncated]

### L3-4e990dfcec  (L3, 2026-09-09, sha 4e990dfcec4b, PR #55968)
TITLE: [ROCm] Bump AITER to v0.1.21.post2 (#55968)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.rocm_base (+1/-1)
LABELS: rocm, ready, ci/build
BODY: Bump AITER from v0.1.21.post1 to v0.1.21.post2

### L3-d8d53f17c2  (L3, 2026-09-09, sha d8d53f17c231, PR #53664)
TITLE: [Rocm][Kimi-k3] Add pipeline_parallel support for the kimik3 model (#53664)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+10/-0); tests/models/test_registry.py (+2/-0)
LABELS: rocm, ready, verified, kimi, k3, kv-cache-manager
DEEP_STUDY: deep-study: this PR was reverted by PR 56429 (confirmed_revert, reason=other)
BODY: ## Summary ⏎  ⏎ - Assert pipeline-parallel capability for both Kimi-K3 model architectures. ⏎ - Derive DCP-local MLA decode sequence lengths when `CommonAttentionMetadata` does not carry them, which keeps PP + DCP decode metadata complete. ⏎  ⏎ ## Scope cleanup ⏎  ⏎ This revision removes the shared per-group cache geometry changes and tests that landed in #53598. It also drops the older ROCm AITER MLA backend changes superseded by #51705. The PR now contains o …[truncated]

### L3-7e95735eb9  (L3, 2026-09-09, sha 7e95735eb90f, PR #55888)
TITLE: [Bugfix] Avoid FlexAttention recompiles when request counts change (#55888)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+18/-11); tests/kernels/test_flex_attention.py (+87/-0)
LABELS: bug
BODY: - Keep the request tensors captured by FlexAttention at their allocated capacity so finishing requests does not change their shapes. ⏎ - Add persistent query-offset and sequence-length buffers and copy active rows into them each step. ⏎ - Preserve mask behavior through single-request batches and CUDA graph replay, with numerical checks against an independent dense reference. ⏎  ⏎ The [E2E Scheduling job in build 12711](https://buildkite.com/vllm/amd- …[truncated]

### L3-719284fe15  (L3, 2026-09-09, sha 719284fe158f, PR #56078)
TITLE: [Core][Model] Unify XD-RoPE into M-RoPE and derive the channel count (#56078)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/core/test_apply_rotary_emb.py (+0/-15); tests/models/transformers/test_backend.py (+19/-0); tests/transformers_utils/test_config.py (+48/-0); tests/v1/core/test_output.py (+0/-18); vllm/config/model.py (+3/-7); vllm/model_executor/layers/rotary_embedding/__init__.py (+0/-13); vllm/model_executor/layers/rotary_embedding/xdrope.py (+0/-172); vllm/model_executor/models/interfaces.py (+2/-50); vllm/model_executor/models/transformers/multimodal.py (+22/-10); vllm/multimodal/utils.py (+3/-4); (+8 more)
LABELS: speculative-decoding, ready, multi-modality, mrv2, scheduler
ISSUES: #55140 [Bug] HunyuanOCR (Transformers backend) crashes on startup: "Expected 4 multimodal RoPE channels, got position_ids with shape (3, 1, N)"
BODY: XD-RoPE was added for the native HunYuan-VL implementation, which has since moved to the Transformers modeling backend. Nothing implements `SupportsXDRoPE` any more, so `uses_xdrope_dim` can only return 0 and every XD-RoPE branch is unreachable; a config that did trip the detection would fail the `supports_xdrope` assertion rather than run. ⏎  ⏎ Transformers has collapsed the distinction upstream too: it renames `xdrope_section` to `mrope_section`  …[truncated]

### L3-42d76ee35a  (L3, 2026-09-09, sha 42d76ee35a2a, PR #56112)
TITLE: [Docs] Fix griffe docstring indentation warning in `SupportsMRoPE` (#56112)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/interfaces.py (+3/-4)
LABELS: build-docs
BODY: ## Purpose ⏎  ⏎ The docs build emits: ⏎  ⏎ ``` ⏎ WARNING - griffe: vllm/model_executor/models/interfaces.py:1786: Confusing indentation for continuation line 13 in docstring, should be 4 * 2 = 8 spaces, not 6 ⏎ ``` ⏎  ⏎ `get_mrope_input_positions` documented its return as a free-form bullet list under `Returns:`, where the wrapped line was indented 6 spaces relative to the section: more than an item (4) but less than a continuation (8), which griffe cannot class …[truncated]

### L3-94848eda60  (L3, 2026-09-09, sha 94848eda600a, PR #55893)
TITLE: [Bugfix] Fix unreachable None guard in Molmo2 get_candidate_target_fps (#55893)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/multimodal/video.py (+3/-2)
LABELS: bug, ready, multi-modality
BODY: ## Purpose ⏎  ⏎ In `Molmo2VideoBackend.get_candidate_target_fps`, the `sampling_fps is None` guard is placed **after** `sampling_fps = int(sampling_fps)`: ⏎  ⏎ ```python ⏎ video_fps = int(video_fps) ⏎ sampling_fps = int(sampling_fps)   # int(None) -> TypeError HERE ⏎ max_fps = int(max_fps) ⏎  ⏎ if sampling_fps is None:           # dead code: never reached when None ⏎     raise ValueError("sampling_fps must be provided") ⏎ ``` ⏎  ⏎ When `sampling_fps is None`, `int(None)`  …[truncated]

### L3-e8064a96d0  (L3, 2026-09-09, sha e8064a96d02d, PR #55745)
TITLE: [Bugfix][V2] record_stream idx_mapping in the PP draft broadcast (#55745)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/pp_utils.py (+3/-1)
LABELS: bug, ready, mrv2
BODY: ## Purpose ⏎  ⏎ `PPHandler.broadcast_drafts` reads a main-stream tensor on the broadcast stream without a `record_stream` guard, so the CUDA caching allocator can recycle that tensor's memory while the gather is still in flight. The recycled indices then land out of bounds: ⏎  ⏎ ``` ⏎ ATen/native/cuda/IndexKernel.cu:111: operator(): block: [0,0,0], thread: [0,0,0] ⏎ Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed. ⏎ ``` ⏎  ⏎ usua …[truncated]

### L3-08b3e67b66  (L3, 2026-09-09, sha 08b3e67b669d, PR #45900)
TITLE: [ROCm][Perf] Fix Qwen3-vLLM audio encoder TP when heads are not divisible by TP size (#45900)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen3_omni_moe_thinker.py (+9/-6)
LABELS: rocm, ready, qwen
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Fix Qwen3-Omni audio encoder init/runtime failure at TP=8 when encoder_attention_heads is not divisible by tensor parallel size. Disable TP on audio encoder attention/FFN layers in that case instead of failing on num_heads // tp_size. ⏎  ⏎ A similar PR that was opened on vllm-omni (https://github.com/vllm-project/vllm-omni/pull/4322#pullrequestreview-4520784172) has already been approved.

### L3-62f3bf58a5  (L3, 2026-09-09, sha 62f3bf58a504, PR #56035)
TITLE: [Bugfix][ROCm][DSv4] Skip launch_pdl=True JIT warmup when PDL is unsupported (#56035)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/models/common/ops/fused_qk_rmsnorm.py (+1/-1); vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py (+1/-1)
LABELS: bug, rocm, ready, deepseek, DSv4
ISSUES: #56030 [Bug][ROCm][DSv4] JIT warmup compiles launch_pdl=True Triton kernels and fails with invalid griddepcontrol on AMD
BODY: ## Summary ⏎ - JIT warmup after #50176 enumerated `launch_pdl=(False, True)` for DeepSeek V4 fused Q/KV RMSNorm and inv-RoPE FP8 quant kernels, even on platforms without PDL. ⏎ - On ROCm that compiles NVIDIA `griddepcontrol` PTX into an hsaco and `ld.lld` fails, so engine startup dies (`#56030`). ⏎ - Only include `launch_pdl=True` in the warmup space when `current_platform.is_arch_support_pdl()` is true. NVIDIA still warms both variants. ⏎  ⏎ Fixes #5 …[truncated]

### L3-dcd544486b  (L3, 2026-09-09, sha dcd544486b7f, PR #55887)
TITLE: [ROCm][Bugfix] Support shared KV prefill in AITER attention (#55887)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+72/-2); tests/kernels/attention/test_rocm_aiter_fa.py (+148/-0)
LABELS: bug, rocm
BODY: - Let AITER attention read K/V from the shared cache when a layer supplies only Q, covering both prefill and chunked prefill. ⏎ - Allocate the gather workspace using each attention group's KV dimensions and gather only the new prefill/extend tokens. ⏎ - Cover NHD and SHUFFLE layouts, FP8 caches, sliding windows, and cache immutability with 18 regression cases. ⏎  ⏎ `git bisect run` from [nightly 12674](https://buildkite.com/vllm/amd-ci/builds/12674)  …[truncated]

### L3-83252ea899  (L3, 2026-09-09, sha 83252ea899c6, PR #52664)
TITLE: [Performance][ROCm]  Integrate aiter indexer scoring and top-k kernels into MiniMax-M3 sparse attention path (#52664)
SOURCES: subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_minimax_m3.py (+52/-0); vllm/models/minimax_m3/amd/indexer_aiter.py (+646/-0); vllm/models/minimax_m3/amd/model.py (+158/-36); vllm/models/minimax_m3/amd/ops/sparse_pa.py (+45/-8); vllm/models/minimax_m3/amd/sparse_attention_msa.py (+152/-5); vllm/models/minimax_m3/common/sparse_attention.py (+24/-9)
LABELS: rocm, ready, verified, minimax
DEEP_STUDY: deep-study performance PR ()
BODY: Integrate aiter indexer scoring and top-k kernels (AITER PR: https://github.com/ROCm/aiter/pull/4787) into MiniMax-M3 sparse attention path.  ⏎  ⏎ Please note that this PR is dependent on these PRs: ⏎  ⏎ - [https://github.com/vllm-project/vllm/pull/52849](https://github.com/vllm-project/vllm/pull/52849) current PR rebased on this PR- since it is dependent on it. ⏎ - [https://github.com/ROCm/aiter/pull/4787](https://github.com/ROCm/aiter/pull/4787) ⏎  ⏎  …[truncated]

### L3-6983a0883d  (L3, 2026-09-09, sha 6983a0883dc8, PR #53602)
TITLE: [ROCm][CI] Split MI300 Distributed Compile by graph partition mode (#53602)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test-amd.yaml (+22/-3)
LABELS: rocm, ci/build
BODY: ## Purpose ⏎  ⏎ `test_tp2_ar_rms.py::test_tp2_ar_rms_fusions` expands to 64 parametrizations (4 model/impl × 4 attention backends × 2 custom-op combos × 2 graph-partition modes), each building a fresh 2-GPU engine. It makes `(MI300) Distributed Compile` the long pole in AMD CI at 77.7 min (build 12767). ⏎  ⏎ Split it along the `inductor_graph_partition` axis, one agent per mode. The `-k` filters match the literal parametrize ids `inductor_partition`  …[truncated]

### L3-474839f846  (L3, 2026-09-09, sha 474839f8469e, PR #55163)
TITLE: [CI/Build] Upload CPU nightly image to Docker Hub (#55163)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/release-pipeline.yaml (+34/-10); .buildkite/scripts/publish-release-images.sh (+2/-2); .buildkite/scripts/push-nightly-builds-cpu.sh (+58/-0)
LABELS: ci/build, cpu, verified
BODY: What changed ⏎  ⏎ Add nightly Buildkite step to upload and published CPU nightly images in release-pipeline.yaml. ⏎ Add a dedicated CPU nightly publish script in push-nightly-builds-cpu.sh. ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-83fe99399e  (L3, 2026-09-09, sha 83fe99399ec0, PR #55713)
TITLE: [Spec Decode] Add NVFP4 DSpark gathered top-k projection (#55713)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/spec_decode/test_dspark_topk.py (+183/-0); vllm/model_executor/layers/quantization/modelopt.py (+24/-0); vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py (+227/-1); vllm/model_executor/models/qwen3_dspark.py (+57/-2)
LABELS: speculative-decoding, ready, qwen, quantization, dflash
BODY: ## Motivation ⏎  ⏎ DSpark's Markov head computes the draft correction by selecting rows from its W2 projection. With a quantized W2 Markov head, those rows are stored as packed NVFP4 values plus per-group/global scales, so they cannot be indexed and used as ordinary dense weights. The selected rows must be gathered and dequantized before the correction matmul. ⏎  ⏎ This is required to run `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark`, w …[truncated]

### L3-8c87c333b8  (L3, 2026-09-09, sha 8c87c333b84c, PR #55499)
TITLE: [Perf] Fix TRTLLM ragged prefill perf regression (#55499)
SOURCES: path_core, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.mla.common_v1
FILES: docker/Dockerfile (+1/-1); docker/versions.json (+1/-1); requirements/cuda.txt (+2/-2); vllm/model_executor/layers/attention/mla_attention.py (+4/-0); vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py (+39/-0); tests/v1/attention/test_mla_context_chunks.py (+4/-0)
LABELS: ready, ci/build, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ There is a perf regression in flashinfer TRTLLM ragged prefill, as described in https://github.com/flashinfer-ai/flashinfer/issues/4928. This PR fixes it by skipping the checks inside the kernel call, avoiding the CUDA synchronization, and instead do the checks at the MLA Prefill metadata preparation so it is done once per forward instead of inside every kernel call. ⏎  ⏎ ### Benchmark ⏎ Fixed ISL=8k OSL=128 Throughput ⏎  ⏎ | Concurrency | …[truncated]

### L3-be1cb9834b  (L3, 2026-09-10, sha be1cb9834b3d, PR #56215)
TITLE: [Kernel] Optional Q-norm in fused DSv4 MLA epilogue; group_size=32 for packed FP8 quant (#56215)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu (+76/-58); csrc/libtorch_stable/ops.h (+3/-3); csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu (+80/-71); csrc/libtorch_stable/torch_bindings.cpp (+4/-3); tests/kernels/quantization/test_per_token_group_quant.py (+5/-0); tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py (+99/-13)
LABELS: deepseek, quantization, DSv4
BODY: ## Purpose ⏎  ⏎ Two independent CUDA changes carved out of #56201 (DeepSeek-V4.1-Flash) so the ⏎ kernel work lands and gets reviewed ahead of the model. Both are behavior- ⏎ preserving for every existing caller. ⏎  ⏎ ### 1. Optional Q-norm in the fused DSv4 MLA epilogue ⏎  ⏎ `csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` ⏎  ⏎ The three DSv4 MLA epilogue ops ⏎ (`fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`, ⏎ `..._full_cache_bf16_insert`, ` …[truncated]

### L3-b47b01cf33  (L3, 2026-09-10, sha b47b01cf3383, PR #56208)
TITLE: [Model][Frontend] Support DeepSeek-V4.1-Flash in Rust and Python frontends (#56208)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: rust/Cargo.lock (+1/-1); rust/Cargo.toml (+1/-1); rust/src/chat/src/backend/hf.rs (+3/-2); rust/src/chat/src/error.rs (+3/-0); rust/src/chat/src/lib.rs (+4/-4); rust/src/chat/src/multimodal.rs (+39/-1); rust/src/chat/src/multimodal/expand.rs (+136/-8); rust/src/chat/src/parser/reasoning/mod.rs (+8/-4); rust/src/chat/src/parser/reasoning/tests.rs (+4/-0); rust/src/chat/src/parser/tool/mod.rs (+7/-4); (+42 more)
LABELS: ready, tool-calling, deepseek, rust, DSv4.1
BODY: ## Purpose ⏎  ⏎ Extract the Rust and Python frontend portions of #56201 onto main for separate review: tokenization/rendering, reasoning and tool parsing, tool grammar, image preprocessing, and config adapters. Pins `llm-multimodal` to `24c676d`; enables roundtrip coverage with `deepseek-ai/DeepSeek-V4.1-Flash`. Model execution and GPU kernels remain in #56201. ⏎  ⏎ ## Test Plan / Results ⏎  ⏎ - `cargo nextest run --locked -p vllm-parser -p vllm-chat --lib`: …[truncated]

### L3-22d95d1adc  (L3, 2026-09-10, sha 22d95d1adc24, PR #55370)
TITLE: [Bugfix] Make `mm_device_do_normalize` encoder-cudagraph safe (#55370)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/multimodal/generation/test_qwen2_vl.py (+6/-2); vllm/model_executor/models/qwen2_5_vl.py (+11/-8); vllm/model_executor/models/qwen2_vl.py (+11/-8)
LABELS: bug, ready, multi-modality, qwen, nvidia
BODY: ## Purpose ⏎  ⏎ Make `mm_device_do_normalize` (introduced in #50411) encoder cudagraph compatible. ⏎  ⏎ Currently `self.input_norm` is applied only in `_process_*_input` (outside the tower), but in the encoder cudagraph path (`encoder_cudagraph_forward`) `self.visual()` is called directly, which results in the normalization skipped under encoder cudagraph on. ⏎  ⏎ - eager mode (default): `self.model.embed_multimodal` -> `self._process_*_input` -> `self …[truncated]

### L3-fe26c705f7  (L3, 2026-09-10, sha fe26c705f709, PR #55921)
TITLE: [Model] Add Bailing V3 VL support (#55921)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/models/supported_models.md (+1/-0); tests/model_executor/test_bailing_mrope.py (+176/-0); tests/models/multimodal/processing/test_bailing_moe_v3_vl.py (+126/-0); tests/models/multimodal/test_mapping.py (+30/-0); tests/models/registry.py (+12/-0); tests/parser/engine/test_ling3.py (+7/-6); tests/transformers_utils/test_bailing_moe_v3_vl_config.py (+159/-0); vllm/config/speculative.py (+12/-1); vllm/model_executor/layers/rotary_embedding/__init__.py (+11/-0); vllm/model_executor/layers/rotary_embedding/bailing_mrope.py (+351/-0); (+9 more)
LABELS: documentation, new-model, speculative-decoding, ready, multi-modality, tool-calling
BODY: ## Purpose ⏎  ⏎ Add native image-text inference support for ⏎ [inclusionAI/Ling-3.0-flash-VL](https://huggingface.co/inclusionAI/Ling-3.0-flash-VL), ⏎ building on the existing Ling 3.0 Flash language model implementation. ⏎  ⏎ ## Implementation ⏎  ⏎ - Add model configuration, registration, and multimodal processing. ⏎ - Reuse the Qwen3 vision encoder and Bailing V3 language model, with a VL projector and Bailing M-RoPE. ⏎ - Support checkpoint weight mappin …[truncated]

### L3-588a813a60  (L3, 2026-09-10, sha 588a813a6044, PR #55213)
TITLE: [ROCm] [BugFix] Fix AITER MXFP4 ASM-GEMM crash on unfused shared experts (#55213)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/quantization/test_rocm_mxfp4.py (+61/-0); vllm/model_executor/kernels/linear/mxfp4/aiter.py (+29/-0); vllm/model_executor/models/qwen3_next.py (+24/-22)
LABELS: bug, rocm, ready, qwen
BODY: ## Summary ⏎ Serving AMD Quark MXFP4 quantized Qwen3.8-Flash-Next checkpoint on gfx950 (MI350x) crashes during weight loading: the shared expert's `down_proj` (in_features=640) violates the AITER ASM FP4 GEMM's scale-swizzle shape assumption. This PR aims to address this issue by adding a per-layer ASM shape guard that flips the layers with crashing shape to the fallback Triton FP4 GEMM. For the case of shared expert fusion enabled, this PR also m …[truncated]

### L3-442d36031c  (L3, 2026-09-10, sha 442d36031ce7, PR #42785)
TITLE: [MM][CG] Enable encoder CUDA Graph for MiniCPM-V (#42785)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/design/cuda_graphs_multimodal.md (+3/-0); examples/generate/multimodal/vision_language_offline.py (+3/-0); tests/models/multimodal/generation/test_vit_cudagraph.py (+65/-0); tests/v1/cudagraph/test_encoder_cudagraph.py (+10/-0); vllm/model_executor/models/cohere_compass.py (+2/-1); vllm/model_executor/models/deepseek_ocr.py (+2/-1); vllm/model_executor/models/ernie45_vl.py (+2/-1); vllm/model_executor/models/gemma3_mm.py (+2/-1); vllm/model_executor/models/gemma4_mm.py (+2/-1); vllm/model_executor/models/glm4_1v.py (+2/-1); (+16 more)
LABELS: documentation, ready, v1, multi-modality, llama, qwen, deepseek, nvidia, kimi, k3
BODY: ## Purpose ⏎ Add encoder CUDA Graph support for MiniCPM-V 2.5, 2.6, 4.0  as part of tracker #38175. This implementation follows the existing workflow introduced in #38061.  ⏎  ⏎ The captured graph covers both the ViT encoder (VPM) and the resampler. ⏎  ⏎ MiniCPM-V 2.0 is not included, as it predates the slice-based vision architecture required by this implementation. ⏎ MiniCPM‑V 4.5 is not included, as its dynamic frame fusion introduces input-dependen …[truncated]

### L3-912d2e79fe  (L3, 2026-09-10, sha 912d2e79fe85, PR #52598)
TITLE: [multimodal][feat] add torchaudio backend to AudioResampler (#52598)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/multimodal/test_audio.py (+81/-3); vllm/multimodal/audio.py (+72/-2); vllm/multimodal/parse.py (+6/-1)
LABELS: ready, multi-modality
BODY: ## Purpose ⏎  ⏎ Add `torchaudio` as a backend of `AudioResampler` and make it the **default** (previously `pyav`), as agreed in discussion with the maintainers: rather than making the resampler configurable at startup, torchaudio becomes the default so no configuration is needed. ⏎  ⏎ What the change does: ⏎  ⏎ - **`resample_audio_torchaudio`**: band-limited sinc interpolation via `torchaudio.transforms.Resample`. Unlike the PyAV path it needs no zero-paddin …[truncated]

### L3-83990f5bcc  (L3, 2026-09-10, sha 83990f5bcc0b, PR #55212)
TITLE: [Bugfix][MRV2] Initialize DCP metadata after batch partitioning (#55212)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+20/-5); tests/v1/spec_decode/test_eagle_draft_attn_metadata.py (+4/-1); tests/v1/worker/test_gpu_input_batch_v2.py (+110/-0); tests/v1/worker/test_gpu_pcp_manager.py (+105/-0); tests/v1/worker/test_gpu_ubatch_slicing.py (+137/-0); vllm/v1/worker/gpu/cp_utils.py (+7/-4); vllm/v1/worker/gpu/cudagraph_utils.py (+10/-12); vllm/v1/worker/gpu/model_runner.py (+18/-16); vllm/v1/worker/gpu/pcp_manager.py (+1/-14); vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py (+11/-13); (+2 more)
LABELS: bug, speculative-decoding, ready, nvidia, mrv2, dflash
BODY: ## TL;DR ⏎  ⏎ Initializes MRV2 DCP-local sequence metadata once from the final runtime `InputBatch`, after PCP partitioning, padding, and dummy-batch construction. This removes duplicated setup paths and prevents missing or stale `dcp_local_seq_lens` in eager, CUDA Graph, and DFlash execution. ⏎  ⏎ ## Purpose ⏎  ⏎ Initialize MRV2 DCP-local sequence lengths from the final runtime `InputBatch`, after PCP partitioning and request padding. ⏎  ⏎ Previously, several p …[truncated]

### L3-2e0ee66cab  (L3, 2026-09-10, sha 2e0ee66cab1e, PR #55819)
TITLE: [Perf] Use UVA-backed contents for MRV2 apply_write (#55819)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/core/test_uva.py (+147/-0); vllm/v1/worker/gpu/buffer_utils.py (+19/-5)
LABELS: ready, mrv2
DEEP_STUDY: deep-study performance PR ()
BODY: PLEASE FILL IN THE PR DESCRIPTION HERE ENSURING ALL CHECKLIST ITEMS (AT THE BOTTOM) HAVE BEEN CONSIDERED. ⏎  ⏎   ## Purpose ⏎  ⏎   In MRV2, `StagedWriteTensor.apply_write()` unconditionally copies staged contents to GPU memory via H2D before the Triton kernel writes them to the target, even when the target is UVA-backed. ⏎  ⏎   This change selects the contents staging location based on the target: GPU-backed targets retain the existing H2D path, while  …[truncated]

### L3-40e6042ec8  (L3, 2026-09-10, sha 40e6042ec83e, PR #54574)
TITLE: [Feature][Spec Decode] MTP with separate (possibly quantized) lm head for nemotron (#54574)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/config/speculative.py (+2/-3); vllm/model_executor/models/nemotron_h_mtp.py (+58/-19)
LABELS: ready, quantization
BODY: ## Purpose ⏎ Enable nemotron_h_mtp to use a separate (possibly quantized) lm head, if one is provided in the MTP checkpoint ⏎ ## Test Plan ⏎ Serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 with an mtp with the following configurations: ⏎ 1. original checkpoint with native MTP ⏎ 2. original checkpoint with external MTP nvidia/Nemotron-3-Super-120B-A12B-BF16-MTPv2 (using a shared lm head) ⏎ 3. original checkpoint with external quantized MTP checkpoi …[truncated]

### L3-6ff479e1f7  (L3, 2026-09-10, sha 6ff479e1f798, PR #55236)
TITLE: [ROCm] Add better kv dtype error discoverability (#55236)
SOURCES: release_notes
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: vllm/platforms/rocm.py (+16/-1)
LABELS: rocm, ready
BODY: ## Purpose ⏎  ⏎ Addresses one of the two issues in #54761. This changes the error message in the event that there is an error due to kv cache dtype being unsupported; it lists explicitly which ones are supported so that the user does not have to guess.

### L3-e6cb56337b  (L3, 2026-09-10, sha e6cb56337b49, PR #56145)
TITLE: [Core] MRV2 support for fast-prefill (#56145)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+7/-2); .buildkite/test_areas/engine.yaml (+1/-1); tests/v1/e2e/general/test_kv_sharing_fast_prefill.py (+15/-1); tests/v1/worker/test_attn_utils.py (+64/-0); tests/v1/worker/test_gpu_model_runner_v2.py (+1/-1); vllm/config/cache.py (+0/-1); vllm/config/vllm.py (+0/-4); vllm/v1/worker/gpu/attn_utils.py (+115/-1); vllm/v1/worker/gpu/input_batch.py (+5/-0); vllm/v1/worker/gpu/model_runner.py (+23/-1); (+2 more)
LABELS: ci/build, mrv2
BODY: 

### L3-7cdd9304ae  (L3, 2026-09-10, sha 7cdd9304ae2e, PR #55531)
TITLE: [KV Connector] Support symmetric DCP disagg for hybrid mamba models (#55531)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/parallel.py (+7/-0); vllm/config/vllm.py (+2/-5); vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py (+5/-3); vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py (+4/-1); vllm/engine/arg_utils.py (+13/-3)
LABELS: ready, kv-connector, mrv2
BODY: ## Purpose ⏎ Support symmetric DCP + dspark disagg for hybrid mamba models (e.g., Kimi K3).  ⏎  ⏎ The NIXL connector automatically sets `cp_kv_cache_interleave_size` to the block size for DCP disagg. However, existing attention backends don't have support for DCP + dspark with `cp_kv_cache_interleave_size > 1`. This PR makes CLI arg `--cp-kv-cache-interleave-size` take priority over the automatic conversion. To run DCP + dspark disagg for the symmet …[truncated]

### L3-6ee5bb0a0b  (L3, 2026-09-10, sha 6ee5bb0a0b3e, PR #54889)
TITLE: [DCP][Kernel][Perf] Fuse the empty-shard LSE mask into the A2A pack kernel (#54889)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/ops/dcp.py (+62/-5); tests/v1/attention/test_dcp_a2a_pack_mask.py (+151/-0)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: # [DCP][Kernel] Fuse the empty-shard LSE mask into the A2A pack kernel ⏎  ⏎ ## Purpose ⏎  ⏎ Under DCP, every MLA layer calls `mask_dcp_empty_shards_` (`vllm/v1/attention/ops/dcp.py`) ⏎ immediately before packing the partial attention output and LSE for the all-to-all ⏎ combine. It sets the LSE of rows whose request holds no local KV shard — and of ⏎ CUDA-graph padding rows — to `-inf`, so those rows carry no weight in the reduction. ⏎  ⏎ Written in eager  …[truncated]

### L3-8359e15aae  (L3, 2026-09-10, sha 8359e15aae32, PR #53695)
TITLE: [ROCm][Feature] Support KV connectors with ROCM_AITER_UNIFIED_ATTN (#53695)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.rocm.aiter_unified
FILES: vllm/v1/attention/backends/rocm_aiter_unified_attn.py (+16/-0); tests/v1/attention/test_rocm_attention_backends_selection.py (+112/-0); tests/v1/kv_connector/unit/test_offloading_connector.py (+15/-1); vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py (+11/-2)
LABELS: rocm, ready, kv-connector, verified
BODY: ## Purpose ⏎  ⏎ It's a block first layout so should be safe to enable. ⏎  ⏎ Previously it was inheriting `False` from `ROCM_ATTN` which is not a block first layout. ⏎  ⏎ ## Test Plan ⏎  ⏎ Run GSM8k on Gemma4 with ROCM_AITER_UNIFIED_ATTN and ⏎ - MoRI connector ⏎ - DecodeBenchConnector ⏎ - NIXL connector ⏎  ⏎ respectively, on MI300. ⏎  ⏎ ### 1. Run Gemma4 with ROCM_AITER_UNIFIED_ATTN and **MoRI** connector. Tested on MI300X. ⏎  ⏎  ⏎ P instance: ⏎ ```shell ⏎ HIP_VISIBL …[truncated]

### L3-93911fcf68  (L3, 2026-09-10, sha 93911fcf6811, PR #53903)
TITLE: [NIXL][PCP] Report replicated-PCP ranks > 0 as done sending instead of hiding them (#53903)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_nixl_connector.py (+8/-3); vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py (+6/-0); vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py (+1/-16); vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py (+5/-0)
LABELS: ready, kv-connector
BODY: ## Purpose ⏎  ⏎ With replicated KV under PCP only PCP rank 0 takes part in NIXL transfers, so `NixlConnector` on ranks > 0 cleared `done_sending` and `get_finished_count()` was lowered to `TP x PP` so the scheduler only waited for rank 0. ⏎  ⏎ That breaks as soon as `NixlConnector` is wrapped in a `MultiConnector` with a sibling that saves on every rank (e.g. `OffloadingConnector`): ⏎  ⏎ - `MultiConnector.get_finished_count()` returns `None` when any c …[truncated]

### L3-48cb12c184  (L3, 2026-09-10, sha 48cb12c184e9, PR #56190)
TITLE: [ROCm][Bugfix] Fix profiler in TheRock image (#56190)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/env_override.py (+74/-11)
LABELS: bug, rocm, ci/build
BODY: ## Purpose ⏎ vLLM's profiling doesn't work reliably in the ROCm 10 image because of subtleties during `import torch` in how TheRock's `rocm_sdk` wheel preloads the ROCm library and how `rocprofiler`'s `find_clients` operates (it's a one-shot `static`). It was observed that some models (e.g. DeepSeek V4) generated no GPU work in traces. This PR works around the issue by trying to preload `libtorch_cpu.so` (which exports `rocprofiler_configure`) as  …[truncated]

### L3-7de70fa7ae  (L3, 2026-09-10, sha 7de70fa7ae9f, PR #56161)
TITLE: [Bugfix][ROCm] Create linear layer biases with `requires_grad=False` (#56161)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/test-amd.yaml (+1/-1); vllm/model_executor/layers/linear.py (+8/-3)
LABELS: bug, rocm, ready, ci/build
BODY: ## Purpose ⏎  ⏎ After upgrading to AITER v0.1.21.post2 on ROCm in https://github.com/vllm-project/vllm/pull/55968, we are seeing the following test failure: ⏎  ⏎ ``` ⏎ pytest -s -v evals/gpt_oss/test_gpqa_correctness.py::test_gpqa_correctness[gpt-oss-20b-rocm-quark-mxfp4-fp8-triton] --config-list-file=configs/models-gfx950.txt ⏎  ⏎ (Worker_TP1 pid=2684) ERROR 09-09 10:03:01 [multiproc_executor.py:1055]   File "/usr/local/lib/python3.12/dist-packages/fly …[truncated]

### L3-c3ccc0e957  (L3, 2026-09-10, sha c3ccc0e957fb, PR #55355)
TITLE: [Model] Add DeepSeek-V4 CPU backend (#55355)
SOURCES: path_core, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/hardware_tests/cpu.yaml (+19/-15); benchmarks/kernels/cpu/benchmark_cpu_fused_moe.py (+3/-1); cmake/cpu_extension.cmake (+8/-1); csrc/cpu/sgl-kernels/common.h (+30/-0); csrc/cpu/sgl-kernels/compressor.cpp (+747/-0); csrc/cpu/sgl-kernels/flash_mla.cpp (+625/-0); csrc/cpu/sgl-kernels/indexer.cpp (+276/-0); csrc/cpu/sgl-kernels/mhc.cpp (+1046/-0); csrc/cpu/sgl-kernels/mla_cache.cpp (+9/-9); csrc/cpu/sgl-kernels/moe.cpp (+3/-4); (+37 more)
LABELS: performance, ci/build, multi-modality, deepseek, cpu, DSv4
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Summary ⏎  ⏎ Adds a CPU backend for DeepSeek-V4, under `vllm/models/deepseek_v4/cpu/`: ⏎  ⏎ - Ported the sparse MLA attention, Lightning indexer, mHC gating, and compressor ⏎   kernels to CPU (AVX512/AMX), under `csrc/cpu/sgl-kernels/` (`flash_mla.cpp`, ⏎   `store_cache.cpp`, `compressor.cpp`, `indexer.cpp`, `paged_mqa_logits.cpp`, ⏎   `topk.cpp`, `mhc.cpp`), replacing the eager PyTorch fallback path. ⏎ - Migrated CPU MoE experts (unquantized, FP8, MXFP4, INT …[truncated]

### L3-7e91760650  (L3, 2026-09-10, sha 7e917606504f, PR #54855)
TITLE: [ROCm][Perf] Route large DSV4 sparse prefill to AITER OPUS (#54855)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+107/-0); tests/kernels/attention/test_rocm_triton_attn_dsv4.py (+211/-0)
LABELS: rocm, ready, verified, DSv4
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Route eligible large DeepSeek V4 sparse-MLA prefill calls on gfx950 to AITER's `pa_sparse_prefill_opus` kernel while keeping the existing Triton implementations as the fallback. ⏎  ⏎ ### End-to-end throughput(8k/1k) ⏎  ⏎ | Max concurrency | Baseline total tok/s | OPUS total tok/s | Gain | ⏎ |---:|---:|---:|---:| ⏎ | 1 | 797.73 | 864.70 | +8.40% | ⏎ | 2 | 1461.54 | 1465.97 | +0.30% | ⏎ | 4 | 2529.05 | 2567.99 | +1.54% | ⏎ | 8 | 4021.02 | 4098 …[truncated]

### L3-7d8d71e989  (L3, 2026-09-10, sha 7d8d71e9897e, PR #43272)
TITLE: [Bugfix] Qwen3-VL(-MoE): pass architectures to with_hf_config for pipeline parallelism (#43272)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen3_vl.py (+4/-1); vllm/model_executor/models/qwen3_vl_moe.py (+4/-1)
LABELS: bug, ready, qwen
ISSUES: #43271 [Bug]: Qwen3-VL-MoE crashes at init with pipeline parallelism ("No model architectures are specified")
BODY: ## Purpose ⏎  ⏎ Fixes #43271. ⏎  ⏎ Qwen3-VL-MoE (and dense Qwen3-VL) crash at engine init with `--pipeline-parallel-size > 1`: ⏎  ⏎ ``` ⏎ pydantic_core._pydantic_core.ValidationError: 1 validation error for VllmConfig ⏎   Value error, No model architectures are specified ⏎ ``` ⏎  ⏎ The inner language model is constructed from `config.text_config`, whose `architectures` is `None` (architectures only exist on the top-level config). `with_hf_config()` does not infer them …[truncated]

### L3-9163190dda  (L3, 2026-09-10, sha 9163190dda00, PR #56098)
TITLE: [ROCm][Bugfix][Perf] Tune multi-stream shared experts use; wvSplitKrc fixes (#56098)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/rocm/skinny_gemms.cu (+216/-32); tests/kernels/quantization/test_rocm_skinny_gemms.py (+97/-0); vllm/distributed/parallel_state.py (+6/-0); vllm/model_executor/layers/fused_moe/runner/moe_runner.py (+0/-13); vllm/model_executor/layers/fused_moe/runner/shared_experts.py (+0/-13); vllm/model_executor/layers/utils.py (+50/-19)
LABELS: bug, rocm
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: After https://github.com/vllm-project/vllm/pull/55099, multi-stream performance in ROCm has markedly improved and allows new use-cases. In particular, the regime in which multi-stream shared-experts decode (implemented in https://github.com/vllm-project/vllm/pull/52033) brings performance advantages has shifted. ⏎  ⏎ ## Purpose ⏎ This PR expands the use of multi-stream shared-experts decode on ROCm by loosening the gate that enables it and tuning it …[truncated]

### L3-9521c60bdc  (L3, 2026-09-10, sha 9521c60bdc0c, PR #54038)
TITLE: [ROCm][Perf] Kimi-K3 Fused kernels for KDA prefill reland (#54038)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+21/-5); csrc/libtorch_stable/kimi_k3/fused_kda_chunk_kernel_rocm.cu (+2346/-0); csrc/libtorch_stable/ops.h (+33/-0); csrc/libtorch_stable/torch_bindings.cpp (+25/-0); tests/models/kimi_k3/test_amd_kda_chunk.py (+859/-0); vllm/engine/arg_utils.py (+17/-5); vllm/models/kimi_k3/amd/kda.py (+44/-23); vllm/models/kimi_k3/amd/ops/kda_chunk.py (+294/-0); vllm/models/kimi_k3/amd/ops/kda_prefill.py (+179/-0)
LABELS: rocm, ready, ci/build, kimi, k3
DEEP_STUDY: deep-study revert record: reland of PR(s) 52606 reason=build_or_dependency || deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ #52606 added a KDA prefill fusion path for ROCm's gfx950, but was reverted in #53294 as its builds failed on gfx90a. This PR adds back the fused path, and guards the kernel builds on gfx950. This PR also adds a few optimizations to further improve the prefill performance, achieving ~5% TTFT reduction compared to the original triton path of the KDA prefill chain. ⏎  ⏎ The bulk of the design follows #52606 in fusing KDA prefill ops into t …[truncated]

### L3-7470082f57  (L3, 2026-09-10, sha 7470082f57a0, PR #51692)
TITLE: [ROCm][Perf] Add bpreshuffled blockscaled fp8 GEMM (#51692)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/_aiter_ops.py (+27/-5); vllm/model_executor/kernels/linear/__init__.py (+4/-0); vllm/model_executor/kernels/linear/scaled_mm/aiter.py (+128/-0)
LABELS: rocm, ready
DEEP_STUDY: deep-study: this PR was reverted by PR 57132 (confirmed_revert, reason=correctness_or_accuracy) || deep-study performance PR (precision_format)
BODY: ## Purpose ⏎  ⏎ Activated when shapes allow for it and configs are tuned. ⏎  ⏎ **Implications (DSv3 1k/1K):**  ⏎ - TP8+DPA: +4-8% QPS ⏎ - TP8+EP: +0-4% QPS ⏎  ⏎ ## Test Plan ⏎  ⏎ Bench serve & accuracy validation with DSv3 on (1) TP8+DPA (2) TP8+EP on 8xMI350. ⏎  ⏎ (note need to run with `VLLM_ROCM_USE_AITER_FP8BMM=0` until https://github.com/vllm-project/vllm/issues/51957 is resolved) ⏎ ```bash ⏎ VLLM_ROCM_USE_AITER=1 \ ⏎ VLLM_ROCM_USE_AITER_FP8BMM=0 \ ⏎ vllm s …[truncated]

### L3-1768273c13  (L3, 2026-09-10, sha 1768273c13d8, PR #55736)
TITLE: [Perf][GLM-5.3-Flash] Decode hot-path cleanups: strided KDA recurrent inputs, NoPE MQA query without concat, no duplicate router GEMM (#55736)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashinfer_sparse
FILES: vllm/model_executor/layers/attention/mla_attention.py (+10/-7); vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py (+5/-1); .buildkite/test_areas/models_basic.yaml (+10/-0); tests/models/glm5next/__init__.py (+2/-0); tests/models/glm5next/test_kda_recurrent.py (+186/-0); vllm/models/glm5next/nvidia/model.py (+4/-4); vllm/models/glm5next/nvidia/ops/third_party/kda/fused_recurrent.py (+35/-9); vllm/models/glm5next/nvidia/ops/third_party/kda/kernels.py (+16/-6)
LABELS: ci/build, nvidia, glm
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ One of three independent GLM-5.3-Flash perf PRs  ⏎  ⏎ Three small, independent decode hot-path cleanups found while profiling GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`, FP8, TP4 on 4x GB300, `FLASHINFER_MLA_SPARSE`): ⏎  ⏎ 1. **KDA decode: read token-strided q/k/v/beta in `fused_recurrent_gated_delta_rule_fwd_kernel`.** The decode path hands the recurrent kernel column slices of the merged `q|k|v` conv output and of the fused `qkvbfg_a` projectio …[truncated]

### L3-b28c3e1568  (L3, 2026-09-10, sha b28c3e1568bf, PR #54713)
TITLE: [BugFix] Retain both replay boundaries so an EAGLE resend of a block-aligned prompt still hits (#54713)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/core/test_prefix_caching.py (+98/-0); tests/v1/core/test_single_type_kv_cache_manager.py (+1/-1); vllm/v1/core/kv_cache_coordinator.py (+19/-20); vllm/v1/core/single_type_kv_cache_manager.py (+12/-10)
LABELS: bug, ready, verified, kv-cache-manager
BODY: ## Purpose ⏎  ⏎ Follow-up to #53945, which fixed Mamba/SWA sparse-retention reuse under EAGLE/MTP by moving the replay boundary to where a lookup lands after the EAGLE drop: ⏎  ⏎ ```python ⏎ aligned = request.num_prompt_tokens // scheduler_block_size * scheduler_block_size ⏎ return max(aligned - scheduler_block_size, 0) ⏎ ``` ⏎  ⏎ That is the right position for a sibling whose prompt merely *starts* with this one. It is not the right position for a resend of the * …[truncated]

### L3-86aca66191  (L3, 2026-09-10, sha 86aca6619161, PR #56159)
TITLE: [Kimi K3 Perf] Avoid KDA mixed-batch gather/scatter, 5.2%~7.7% E2E Throughput Improvement (#56159)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/kimi_k3/test_kda.py (+7/-0); tests/models/kimi_k3/test_kda_metadata.py (+4/-0); vllm/models/kimi_k3/nvidia/kda.py (+33/-3); vllm/models/kimi_k3/nvidia/kda_metadata.py (+13/-0); vllm/models/kimi_k3/nvidia/ops/third_party/kda/chunk.py (+6/-1); vllm/models/kimi_k3/nvidia/ops/third_party/kda/fused_recurrent.py (+3/-1)
LABELS: ready, kimi, k3
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Part of https://github.com/vllm-project/vllm/issues/50587 ⏎  ⏎  ⏎ ```bash ⏎ # Example packed batch: ⏎ Token order: [N0, N1, S0, S1, S2] ⏎               └─ N ─┘  └──── S ────┘ ⏎  ⏎ Packed QKV / gate / beta ⏎           | ⏎           +-- index_select(non-spec) × 3 --> non-spec KDA --+ ⏎           |                                                  | ⏎           +-- index_select(spec)     × 3 --> spec KDA -------+ ⏎                                    …[truncated]

### L3-a36dfc93cb  (L3, 2026-09-10, sha a36dfc93cb1e, PR #56310)
TITLE: [Bugfix][Multimodal] Restore cached audio inputs with UUIDs (#56310)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/entrypoints/unit_tests/test_chat_utils.py (+6/-2); tests/multimodal/test_parse.py (+18/-0); vllm/entrypoints/chat_utils.py (+1/-1); vllm/multimodal/parse.py (+7/-2)
LABELS: bug, frontend, ready, multi-modality
BODY: ## Purpose ⏎  ⏎ The [documented cached-audio request](https://docs.vllm.ai/en/latest/features/multimodal_inputs/#cached-inputs) fails even after the audio has been cached: ⏎  ⏎ ```json ⏎ {"type": "input_audio", "input_audio": null, "uuid": "clip-a"} ⏎ ``` ⏎  ⏎ Chat parsing rejects this as a missing `type`. Using `{}` gets past that check, but the resulting `audio: [None]` then hits `assert_never` in the audio data parser before cache lookup. ⏎  ⏎ This preserves the  …[truncated]

### L3-828f4f19b4  (L3, 2026-09-10, sha 828f4f19b4d8, PR #55239)
TITLE: [ROCm][Bugfix] Route GLM-5.3-Flash MTP through ragged sparse MLA (#55239)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+7/-4); tests/v1/attention/test_rocm_glm5next_sparse.py (+4/-1)
LABELS: bug, rocm, ready, glm
BODY: ## Purpose ⏎  ⏎ Fix ROCm sparse-MLA dispatch for GLM-5.3-Flash during multi-token speculative decoding. ⏎  ⏎ The existing selector routes ordinary BF16 rope-free decode through the Triton ragged kernel, but excludes speculative verification rows when `max_query_len > 1` or `num_decode_tokens != num_decodes`. Those rows consequently fall through to an incompatible AITER kernel and **produce corrupted output leading to 0% correct on GSM8k.** ⏎  ⏎ This im …[truncated]

### L3-9b959b8657  (L3, 2026-09-10, sha 9b959b86577c, PR #56228)
TITLE: [Model] DeepSeek-V4.1-Flash Model Definitions (#56228)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.dispatch.registry
FILES: vllm/models/deepseek_v4_1/nvidia/flashmla.py (+387/-0); tests/models/test_deepseek_v4_mega_moe.py (+47/-11); tests/parser/engine/trace_builder.py (+40/-17); vllm/model_executor/kernels/attention/dsa/candidate_blocks.py (+234/-0); vllm/model_executor/kernels/linear/__init__.py (+15/-3); vllm/model_executor/kernels/linear/mxfp8/Mxfp8LinearKernel.py (+1/-1); vllm/model_executor/kernels/linear/mxfp8/deep_gemm.py (+128/-0); vllm/model_executor/kernels/linear/mxfp8/flashinfer.py (+38/-14); vllm/model_executor/kernels/mhc/tilelang.py (+200/-0); vllm/model_executor/kernels/mhc/tilelang_kernels.py (+67/-18); (+37 more)
LABELS: new-model, ready, tool-calling, deepseek, nvidia, quantization, DSv4, DSv4.1
BODY: Split out from #56214 — this PR includes **only** the changes under `vllm/models/`: ⏎  ⏎ - Adds the new `vllm/models/deepseek_v4_1/` model definition package (nvidia, amd, and common code paths) ⏎ - Includes the associated modifications to `vllm/models/deepseek_v4/` ⏎  ⏎ All other changes from #56214 (kernels, rust parser, config, tests, etc.) are intentionally excluded. ⏎  ⏎ **Non-duplication:** This is not a competing implementation — it is a scoped subset o …[truncated]

### L3-07b7553465  (L3, 2026-09-10, sha 07b75534651a, PR #54736)
TITLE: [Feature][SimpleCPU] Load fine-grained hybrid prefix hits (#54736)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/simple_kv_offload/test_kv_events.py (+9/-40); tests/v1/simple_kv_offload/test_scheduler.py (+672/-0); vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py (+7/-0); vllm/v1/core/kv_cache_coordinator.py (+12/-4); vllm/v1/core/single_type_kv_cache_manager.py (+16/-0); vllm/v1/simple_kv_offload/manager.py (+266/-66)
LABELS: ready, kv-connector, verified, kv-cache-manager
BODY: ## Purpose ⏎  ⏎ On hybrid models, SimpleCPU offload cannot serve a fine-grained external prefix hit reliably: ⏎  ⏎ - Mamba `align` mode nulls interior states and relocates speculative blocks in place, so its block table cannot be resolved by a monotonically advancing positional cursor. The connector must consume the exact boundary handoffs already published by the KV cache manager. ⏎ - A prompt can end on a hash boundary that is not a full-attention e …[truncated]

### L3-980c16c8e4  (L3, 2026-09-10, sha 980c16c8e4c6, PR #56107)
TITLE: [PCP][Spec Decode] Adds PCP support for single-module MTP and replicated DSpark. (#56107)
SOURCES: symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/spec_decode/test_eagle_draft_attn_metadata.py (+1/-0); tests/v1/worker/test_gpu_autoregressive_speculator.py (+1/-0); tests/v1/worker/test_gpu_pcp_manager.py (+1/-10); vllm/model_executor/layers/sparse_attn_indexer.py (+3/-2); vllm/v1/worker/cp_utils.py (+7/-3); vllm/v1/worker/gpu/model_runner.py (+12/-4); vllm/v1/worker/gpu/pcp_manager.py (+84/-53); vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py (+29/-5); vllm/v1/worker/gpu/spec_decode/dflash/speculator.py (+12/-0); vllm/v1/worker/gpu/spec_decode/mtp/speculator.py (+2/-1); (+1 more)
LABELS: speculative-decoding, ready, mrv2, dflash
BODY: ## Summary ⏎  ⏎ Adds V2 model-runner PCP support for single-module MTP and replicated DSpark. ⏎  ⏎ - MTP reuses the target PCP-local batch for its initial draft forward, then refreshes global-order block tables for subsequent replicated drafts. ⏎ - DSpark stays replicated while the target is PCP-sharded. ⏎ - PIECEWISE CUDA graphs are supported; full graphs remain rejected. ⏎  ⏎ This consolidates #53653 and #53660 on current `main` with a smaller implementation. ⏎  …[truncated]

### L3-fbf51c7026  (L3, 2026-09-10, sha fbf51c70269b, PR #55864)
TITLE: [Bugfix] Fix FlashInfer KV sharing with omitted K/V (#55864)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/model_executor/models/gemma4_mtp.py (+2/-10); vllm/v1/attention/backends/flashinfer.py (+12/-8)
LABELS: bug, ready, nvidia, verified
BODY: This is a narrow forward fix for the FlashInfer regression covered by #55789, ⏎ preserving #54917 instead of reverting it. ⏎  ⏎  ⏎  ⏎ ## Purpose ⏎  ⏎ [#54917](https://github.com/vllm-project/vllm/pull/54917) avoids computing K/V projections for Gemma4 KV-sharing layers and invokes attention with `key=None` and `value=None`. FlashInfer still sliced these tensors unconditionally while removing CUDA Graph padding, causing startup to fail with: ⏎ ```text ⏎ Ty …[truncated]

### L3-84030bbe3d  (L3, 2026-09-11, sha 84030bbe3d74, PR #49675)
TITLE: [Bugfix][Core] Stop zero-progress preemption cascades for deferred KV frees (#49675)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/core/test_async_scheduler.py (+39/-15); tests/v1/core/test_deferred_block_free.py (+72/-2); tests/v1/core/utils.py (+3/-0); vllm/v1/core/sched/scheduler.py (+16/-10)
LABELS: bug, ready, v1, scheduler
ISSUES: #49674 [Bug]: Deferred KV block frees cause zero-progress preemption cascades with async KV consumers
BODY: Fixes #49674 ⏎  ⏎ ## Scope ⏎  ⏎ This PR fixes one known zero-progress transition in the running-request ⏎ allocation retry. It does not change the deferred-free safety fence, model ⏎ output semantics, or FCFS/PRIORITY victim-selection policy. ⏎  ⏎ ## Change ⏎  ⏎ The scheduler now uses the existing deferred-free conditions through one ⏎ predicate. Before mutating the running queue, it applies that predicate to the ⏎ victim already selected by FCFS/PRIORITY. If the victim …[truncated]

### L3-d0dfe587d5  (L3, 2026-09-11, sha d0dfe587d5b2, PR #55095)
TITLE: [Bugfix] Fall back to full decode graphs for noncompiled models (#55095)
SOURCES: release_notes
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: tests/test_config.py (+122/-0); tests/v1/spec_decode/test_adaptive_verification.py (+22/-0); vllm/config/compilation.py (+22/-0); vllm/config/vllm.py (+33/-17); vllm/platforms/rocm.py (+17/-0); vllm/v1/worker/gpu/model_runner.py (+10/-1); vllm/v1/worker/gpu/spec_decode/adaptive_verification.py (+11/-0)
LABELS: bug, rocm, speculative-decoding, needs-rebase, mrv2
BODY: - Apply the fallback only to platform-selected model paths without an active torch-compile boundary. ⏎ - Preserve ROCm's compiled MRV1 defaults and the generic compiled implementations used by DeepSeek V3.2 and GLM DSA models. ⏎ - Default unset graph modes to `FULL_DECODE_ONLY` at O2/O3 and `NONE` at O0/O1 when breakable graphs are unavailable. ⏎ - Default DeepSeek V4 MRV2 to `NONE` on gfx950 because full-decode capture has an unresolved accuracy risk  …[truncated]

### L3-1cf6555214  (L3, 2026-09-11, sha 1cf655521444, PR #56138)
TITLE: [Bugfix] Pin EPLB and MLA host-to-device transfer buffers (#56138)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention/sparse_mla_attention.py (+12/-4); tests/distributed/test_eplb_execute.py (+25/-0); tests/v1/attention/test_mla_context_chunks.py (+40/-0); vllm/distributed/eplb/eplb_communicator.py (+4/-1)
LABELS: bug
BODY: Main CI build 87842 fails both Qwen3 Sync EPLB accuracy jobs and the TP1/PCP4 evaluation when strict GPU synchronization checks detect nonblocking H2D copies from pageable CPU memory. ⏎  ⏎ Pin Gloo receive staging buffers and allocate sparse MLA context lengths directly in pinned CPU memory with torch.subtract(out=...), matching the dense MLA caller. Preserve the strict checks. Add focused CUDA regressions that exercise the actual transfers and verif …[truncated]

### L3-e77daef89e  (L3, 2026-09-11, sha e77daef89e18, PR #56214)
TITLE: [Model] Support DeepSeek-V4.1-Flash (#56214)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: .buildkite/test_areas/kernels.yaml (+3/-0); tests/fusion/test_quant_activation_contract.py (+8/-0); tests/kernels/attention/test_rocm_triton_attn_dsv4.py (+25/-0); tests/kernels/core/test_fused_q_kv_rmsnorm.py (+189/-0); tests/kernels/quantization/test_rocm_mxfp8_linear.py (+144/-0); tests/kernels/test_compressor_kv_cache.py (+432/-0); tests/kernels/test_engram.py (+778/-0); tests/kernels/test_fused_indexer_q_rope_quant.py (+359/-9); tests/kernels/test_fused_inv_rope_fp8_quant.py (+88/-17); tests/kernels/test_mhc_kernels.py (+285/-1); (+39 more)
LABELS: new-model, rocm, speculative-decoding, torch.compile, ci/build, multi-modality, tool-calling, deepseek, kv-connector, nvidia
BODY: 

### L3-5ff50f3996  (L3, 2026-09-11, sha 5ff50f3996d5, PR #56385)
TITLE: [Bugfix][MM] Fix swapped H/W in dummy video profiling inputs (#56385)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/multimodal/processing/dummy_inputs.py (+1/-1)
LABELS: bug, ready, multi-modality
BODY: ## Purpose ⏎  ⏎ `BaseDummyInputsBuilder._get_dummy_videos` builds the dummy profiling video with **width and height swapped**: ⏎  ⏎ ```python ⏎ # vllm/multimodal/processing/dummy_inputs.py (before) ⏎ video = np.full((num_frames, width, height, 3), 255, dtype=np.uint8) ⏎ ``` ⏎  ⏎ Every other video code path in `vllm/multimodal/` uses the `(T, H, W, C)` layout, e.g. `rescale_video_size`: ⏎  ⏎ ```python ⏎ # vllm/multimodal/video.py ⏎ _, height, width, _ = frames.shape ⏎ ``` ⏎  ⏎ S …[truncated]

### L3-a5b714eee4  (L3, 2026-09-11, sha a5b714eee4ef, PR #53699)
TITLE: [Bugfix] Fix Qwen3-VL and Cosmos3-Edge text architectures for CPU and pipeline parallelism (#53699)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/cosmos3_edge.py (+3/-1)
LABELS: bug, ready, qwen
BODY: ## Purpose ⏎  ⏎ Fix engine initialization failures for Qwen3-VL, Qwen3-VL-MoE, and ⏎ Cosmos3-Edge when vLLM constructs an inner language-model configuration from a ⏎ nested text config. ⏎  ⏎ This PR builds on the Qwen3-VL fix in #43272 and extends the same model-side ⏎ architecture handling to Cosmos3-Edge. ⏎  ⏎ The issue is exposed by: ⏎  ⏎ - CPU initialization, including `pipeline_parallel_size=1`. ⏎ - Pipeline-parallel capability validation when ⏎   `pipel …[truncated]

### L3-4b839c3378  (L3, 2026-09-11, sha 4b839c3378c6, PR #50439)
TITLE: [Attention] Extend XQA decode support on SM90 (#50439)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+87/-92); tests/v1/attention/test_attention_backends.py (+147/-13); tests/v1/attention/test_flashinfer_dcp_spec_reorder.py (+5/-0)
LABELS: ready, ci/build, v1, nvidia
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Summary ⏎  ⏎ Follow up on #49718, which added the dedicated FlashInfer XQA path for SM12x. ⏎  ⏎ - Use FlashInfer's dedicated XQA API on SM90 for uniform and ragged speculative decode. ⏎ - Support non-causal draft attention and attention sinks on the SM90 XQA path. ⏎ - Remove the obsolete SM90 sliding-window guard; current `main` pins FlashInfer 0.6.17, which contains the upstream fix. ⏎ - Preserve padded query/output rows for single-token full CUDA g …[truncated]

### L3-5b6cf93e8e  (L3, 2026-09-11, sha 5b6cf93e8e22, PR #54968)
TITLE: [XPU] Add forward_xpu to Mixer2RMSNormGated and FusedRMSNormGated (#54968)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/third_party/flash_linear_attention/ops/kda.py (+16/-0); vllm/model_executor/layers/mamba/mamba_mixer2.py (+7/-0)
LABELS: intel-gpu, verified
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ Mixer2RMSNormGated and FusedRMSNormGated define forward_cuda but no forward_xpu, so on XPU they fall to forward_native. Both forward_cuda paths call in-tree Triton (rms_norm_gated), which is portable, so XPU can use it directly. This only affects eager mode execution as TC mode will run forward_native by default. ⏎  ⏎ **Affected architectures (depends on TP size and model variant)**: FalconH1ForCausalLM, GraniteMoeHybridForCausalLM, M …[truncated]

### L3-1e1060f998  (L3, 2026-09-11, sha 1e1060f9988f, PR #55356)
TITLE: [Kimi Perf] Group fp8 mla cahche insertion, 4~6x kernel level performance improvement for small batch (#55356)
SOURCES: path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: vllm/_custom_ops.py (+4/-0); csrc/libtorch_stable/cache_kernels.cu (+92/-25); csrc/libtorch_stable/ops.h (+6/-6); csrc/libtorch_stable/torch_bindings.cpp (+5/-3); tests/kernels/attention/test_cache.py (+88/-0); vllm/models/kimi_k3/nvidia/dspark_mla.py (+21/-5)
LABELS: ready, dflash, kimi, k3
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ Group fp8 mla cahche insertion ⏎  ⏎ Before: ⏎  ⏎ ```bash ⏎ CUDA launch 1：layer 0 ⏎ CUDA launch 2：layer 1 ⏎ CUDA launch 3：layer 2 ⏎ CUDA launch 4：layer 3 ⏎ CUDA launch 5：layer 4 ⏎ ``` ⏎  ⏎ Now ⏎ ```bash ⏎ One grouped kernel launch ⏎   ├─ token 0, layer 0 ⏎   ├─ token 0, layer 1 ⏎   ├─ ... ⏎   └─ ... ⏎ ``` ⏎  ⏎ So basically combining 5 layers together, saving a lot of launch time ⏎  ⏎ ## Test ⏎  ⏎ Acc covered in current unit test ⏎  ⏎ Perf can be seen in this A …[truncated]

### L3-5fe77aecfc  (L3, 2026-09-11, sha 5fe77aecfc76, PR #55353)
TITLE: [Deprecation] Deprecate items scheduled for 0.29 (#55353)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.rocm_aiter_sparse, L3.dispatch.abstract_interface, L3.platform.rocm_selection
FILES: vllm/model_executor/layers/attention/cross_attention.py (+1/-1); vllm/v1/attention/backend.py (+1/-41); vllm/v1/attention/backends/flashinfer.py (+1/-1); vllm/v1/attention/backends/mla/indexer.py (+2/-2); vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+5/-2); vllm/v1/attention/backends/utils.py (+2/-8); benchmarks/attention_benchmarks/mla_runner.py (+0/-6); tests/engine/test_arg_utils.py (+0/-18); tests/kernels/attention/test_rocm_aiter_mla_sparse_metadata_sync.py (+2/-2); tests/kernels/test_compressor_kv_cache.py (+6/-2); (+18 more)
LABELS: performance, rocm, speculative-decoding, ready, deepseek, kv-connector, nvidia, DSv4, dflash
BODY: ## Purpose ⏎  ⏎ Deprecate items scheduled for 0.29

### L3-127143e27f  (L3, 2026-09-11, sha 127143e27f2b, PR #56429)
TITLE: Revert "[Rocm][Kimi-k3] Fix pipeline_parallel support for the kimik3 DCP mode  (#53664)" (#56429)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+0/-10); tests/models/test_registry.py (+0/-2)
LABELS: rocm, kimi, k3
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 53664 reason=other
BODY: ## Purpose ⏎  ⏎ This PR reverts #53664 ("[Rocm][Kimi-k3] Fix pipeline_parallel support for the kimik3 DCP mode"). That PR added a fallback in `MLAAttentionLayer.build()` that derives `dcp_local_seq_lens` from `seq_lens` whenever `CommonAttentionMetadata` does not carry it, together with an assert and two test-registry entries. The fallback is dead code: in vLLM's in-tree paths, `dcp_local_seq_lens` is always populated before the MLA decode path run …[truncated]

### L3-9dd969da09  (L3, 2026-09-11, sha 9dd969da096e, PR #55107)
TITLE: [Model][ROCm] Enable DeepSeek V4 Vision (#55107)
SOURCES: release_notes
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: .buildkite/test_areas/models_basic.yaml (+4/-2); tests/kernels/attention/test_rocm_triton_attn_dsv4.py (+95/-0); tests/models/multimodal/processing/test_tensor_schema.py (+4/-3); tests/models/test_deepseek_v4_vl_rocm.py (+458/-0); tests/models/test_initialization.py (+2/-2); tests/models/test_registry.py (+4/-3); tests/test_config.py (+39/-0); tests/v1/spec_decode/test_adaptive_verification.py (+1/-0); vllm/models/deepseek_v4/__init__.py (+4/-4); vllm/models/deepseek_v4/amd/model.py (+23/-1); (+7 more)
LABELS: rocm, speculative-decoding, ci/build, multi-modality, deepseek, mrv2, DSv4
BODY: - Enable `DeepseekV4ForConditionalGeneration` on ROCm by moving the platform-neutral wrapper to `common/`, retaining the NVIDIA compatibility shim and unsupported XPU stub, and enabling ROCm registry, dummy-init, and tensor-schema paths. ⏎ - Build checkpoint mappings from the active text backend, preserving NVIDIA self-finalization while deferring and idempotently applying ROCm finalization after generic per-layer quantization. ⏎ - Route image sent …[truncated]

### L3-8c1d1c2974  (L3, 2026-09-11, sha 8c1d1c2974ee, PR #55127)
TITLE: [Misc] Log FlashInfer allreduce workspace init failure as error (#55127)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/distributed/device_communicators/flashinfer_all_reduce.py (+2/-2); vllm/logger.py (+16/-0)
LABELS: ready, nvidia
BODY: When the FlashInfer allreduce workspace fails to initialize and the backend fallback is exhausted, the only signal was a `WARNING`. The downstream failure is typically a bare `assert workspace is not None` deep in the fusion pass (`allreduce_rms_fusion.py`), far from the root cause — which surfaces only through this once-deduped log line. Severity should reflect the outcome, so the definitive failure is now logged at error level. ⏎  ⏎ Attempt-phase …[truncated]

### L3-b4da4d17ae  (L3, 2026-09-11, sha b4da4d17ae0c, PR #56433)
TITLE: [ROCm][Bugfix] Fix AITER preshuffled FP8 block-scale kernel (#56433)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+3/-4); vllm/model_executor/kernels/linear/scaled_mm/aiter.py (+49/-7); vllm/model_executor/layers/linear.py (+3/-3); vllm/models/deepseek_v4/amd/model.py (+11/-6); vllm/models/deepseek_v4/amd/rocm.py (+54/-16)
LABELS: bug, rocm, ready, deepseek, DSv4
DEEP_STUDY: deep-study: this PR was reverted by PR 57132 (confirmed_revert, reason=correctness_or_accuracy)
BODY: ## Purpose ⏎  ⏎ This PR fixes breakages in MLA models on ROCm. ⏎  ⏎ https://github.com/vllm-project/vllm/pull/51692 added AiterPreshuffledFp8BlockScaledMMKernel first in the ROCm block-scaled priority list, so DeepSeek-V4/V3/Kimi-K2/many MLA models select it. Four things there lead to incorrect results: ⏎ - It reads the activation scale as column-major, while pre-quantizing callers emit row-major scales. Both unfortunately have the same shape `(M, N / …[truncated]

### L3-8a7f98c8c3  (L3, 2026-09-11, sha 8a7f98c8c3ae, PR #53280)
TITLE: [Kernel][MoE] Optimize batched_moe_align_block_size with cooperative writes (#53280)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: benchmarks/kernels/benchmark_moe_align_block_size.py (+82/-1); csrc/libtorch_stable/moe/moe_align_sum_kernels.cu (+60/-18); tests/kernels/moe/test_moe_align_block_size.py (+12/-8)
LABELS: performance, ready
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Summary ⏎  ⏎ - parallelize batched MoE token and block-ID output writes across cooperative thread groups ⏎ - retain the original serial specialization for small capacities and high batch counts ⏎ - expand correctness coverage and add a batched alignment benchmark matrix ⏎  ⏎ The production caller is currently `BatchedMarlinExperts`, used with the batched activation format for supported expert-parallel configurations. ⏎  ⏎ ## Why this is not a duplicate ⏎  ⏎ PRs # …[truncated]

### L3-295ac4e52e  (L3, 2026-09-11, sha 295ac4e52e8a, PR #55326)
TITLE: [Bugfix][Multimodal] Parse decoded video frame lists as a single video (#55326)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/multimodal/test_parse.py (+17/-0); vllm/multimodal/parse.py (+8/-2)
LABELS: bug, ready, multi-modality
BODY: ## Purpose ⏎  ⏎ Fix the multimodal parser misclassifying a list of decoded NumPy or PyTorch ⏎ frames as multiple videos. ⏎  ⏎ ### Code/documentation mismatch ⏎  ⏎ The public contract and the parser currently disagree: ⏎  ⏎ * [`docs/features/multimodal_inputs.md`](https://github.com/vllm-project/vllm/blob/main/docs/features/multimodal_inputs.md#video-inputs) ⏎   explicitly says that a list of NumPy arrays can be passed directly to the ⏎   `video` field, and  …[truncated]

### L3-89dbb26445  (L3, 2026-09-11, sha 89dbb2644552, PR #56447)
TITLE: [Bugfix] Fix GLM-OCR MTP position masking during CUDA graph capture (#56447)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/glm_ocr_mtp.py (+2/-1)
LABELS: bug, ready, nvidia, glm
BODY: Handle both one-dimensional runner positions and two-dimensional MRoPE positions with a broadcast masked fill. Add CUDA graph replay coverage for per-token masking. ⏎  ⏎ Validated: isolated GPU regression cases passed for both position layouts; five deterministic OCR outputs matched the non-MTP baseline, including four concurrent requests.

### L3-ae48466cf3  (L3, 2026-09-11, sha ae48466cf33b, PR #48247)
TITLE: [Perf][ROCm] Add AITER custom AG/RS (DP only) (#48247)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: benchmarks/kernels/benchmark_device_communicators.py (+204/-51); tests/distributed/test_comm_ops.py (+15/-0); tests/distributed/test_rocm_aiter_custom_ar.py (+171/-1); tests/utils.py (+52/-10); vllm/distributed/device_communicators/aiter_custom_all_reduce.py (+14/-0); vllm/distributed/device_communicators/cuda_communicator.py (+83/-4); vllm/distributed/parallel_state.py (+12/-10); vllm/envs.py (+1/-1)
LABELS: performance, rocm, ready, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Part of #48255. ⏎  ⏎ Only activated under uniform batches and on the **DP process group** (not TP). **Enabled by default when using DP attention + TP experts**. ⏎  ⏎ **Note**: this does not touch any collectives on the TP or EP groups, so EP a2a's or TP AR's or sequence parallel a2a (it uses the EP group) is not affected. ⏎  ⏎ Disable with `VLLM_ROCM_USE_AITER_CUSTOM_AR=0`. ⏎  ⏎ **Perf gain:** **~3% improved TPOT on 1k/1k** (1-256 conc) ⏎  ⏎  …[truncated]

### L3-5392fbca2a  (L3, 2026-09-11, sha 5392fbca2a45, PR #56459)
TITLE: [ROCm][Docker] Pin AINIC apt repo to snapshot 1.117.5-a-77 (#56459)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.rocm (+1/-1); docker/Dockerfile.rocm_gfx1250 (+1/-1); docs/features/moriio_connector_usage.md (+1/-1)
LABELS: documentation, rocm, ready, ci/build
BODY: ## Purpose ⏎ The AMD Model Executor job fails during environment setup: ⏎  ⏎ ``` ⏎ E: Failed to fetch .../amdainic/pensando/ubuntu/1.117.5/dists/jammy/main/binary-amd64/Packages.gz  ⏎ File has unexpected size (3363 != 3397). Mirror sync in progress? ⏎ ``` ⏎  ⏎ `AINIC_VERSION=1.117.5`, set in #54112, points at a mutable path whose contents are updated in place. The index files behind `repo.radeon.com` are cached per file, so a client can end up pairing an …[truncated]

### L3-9dcf6bf344  (L3, 2026-09-11, sha 9dcf6bf344ca, PR #56181)
TITLE: [BugFix] Fix DP token padding in dflash attention metadata (#56181)
SOURCES: path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py (+6/-8); vllm/v1/worker/gpu/spec_decode/dflash/speculator.py (+3/-33); vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py (+2/-3); vllm/v1/worker/gpu/spec_decode/speculator.py (+45/-30); tests/v1/spec_decode/test_eagle_draft_attn_metadata.py (+41/-23)
LABELS: bug, speculative-decoding, ready, mrv2, dflash
BODY: ## Summary ⏎ This PR follows up on the fix in https://github.com/vllm-project/vllm/pull/55458, but for dflash-derived speculators. While serving DFlash/DSpark models with DP > 1, the extra padding tokens from the DP sync can cause a mismatch between the number of query tokens reported in `query_start_loc_cpu` and the number reported by `num_tokens` in the attention metadata. When this happens, it can trigger the following assertion for the FlashIn …[truncated]

### L3-fadfe1c7d4  (L3, 2026-09-11, sha fadfe1c7d4df, PR #55450)
TITLE: [Bugfix][Core] Retire Mamba states across null gaps (#55450)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/core/test_single_type_kv_cache_manager.py (+76/-0); tests/v1/e2e/general/test_mamba_prefix_cache.py (+1/-1); vllm/v1/core/single_type_kv_cache_manager.py (+23/-0)
LABELS: bug, ready, kv-cache-manager
BODY: ## Purpose ⏎  ⏎ Fix align-mode Mamba retirement stopping at null gaps and leaking older states. Track the retired prefix to avoid rescanning it. ⏎  ⏎ Split from #55435. #54076 changes chunk splitting; #53803 changes checkpoint retention. Neither fixes this leak. AI-assisted contribution. ⏎  ⏎ ## Test Plan ⏎  ⏎ ```bash ⏎ .venv/bin/python -m pytest \ ⏎   tests/v1/core/test_single_type_kv_cache_manager.py \ ⏎   tests/v1/core/test_deferred_block_free.py \ ⏎   te …[truncated]

### L3-e3f755b732  (L3, 2026-09-11, sha e3f755b73231, PR #55352)
TITLE: [CPU] Speedup LM Head on Arm CPUs (#55352)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/cpu/dnnl_helper.cpp (+7/-4); csrc/cpu/dnnl_helper.h (+3/-0); csrc/cpu/dnnl_kernels.cpp (+26/-2); tests/kernels/test_onednn.py (+1/-0)
LABELS: cpu
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ LM Head takes around 11-14% of runtime for quantized models These layers currently fallback to a sub-optimal kernel in ACL. The most optimal kernel only supports bf16 src and wei tensors with fp32 dst. ⏎  ⏎ We fix this by doing the oneDNN matmul with bf16:bf16:f32 problem desc on AArch64 for BF16 layers with N >= 64k. ⏎  ⏎ Code path for other backends is not changed. ⏎  ⏎ This improves throughput of llama-3.1-8b.w8a8 on 32 Neoverse-V3 cor …[truncated]

### L3-6fe67cbbf3  (L3, 2026-09-11, sha 6fe67cbbf3e4, PR #46994)
TITLE: [Spec][V2] Support MTP speculative decoding under pipeline parallelism (#46994)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.flashinfer_sparse, L3.mla.rocm_aiter_sparse
FILES: vllm/model_executor/layers/attention/sparse_mla_attention.py (+35/-9); vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py (+7/-6); vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+7/-7); vllm/v1/attention/backends/mla/xpu_mla_sparse.py (+5/-7); tests/v1/attention/test_sparse_mla_backends.py (+45/-0); tests/v1/e2e/spec_decode/test_mtp_parallel_load.py (+1/-8); vllm/model_executor/models/deepseek_mtp.py (+16/-2); vllm/model_executor/models/qwen3_5_mtp.py (+9/-3); vllm/models/deepseek_v32/amd/mtp.py (+16/-2); vllm/models/deepseek_v32/nvidia/model.py (+3/-2); (+2 more)
LABELS: rocm, intel-gpu, speculative-decoding, ready, needs-rebase, v1, qwen, deepseek, nvidia, mrv2
BODY: ## Purpose ⏎  ⏎ MTP speculative decoding does not work under pipeline parallelism on the V2 model runner. #50514 ⏎ landed the generic PP spec-decode transport; this PR covers the MTP-specific gaps that remain on ⏎ the PP>1 path. ⏎  ⏎ **Depends on #55745** — a one-line `record_stream` guard in #50514's `broadcast_drafts`, without ⏎ which this path trips a device-side `IndexKernel` assert under async launches. It is the bottom ⏎ commit here and can be dropped once …[truncated]

### L3-dc07f1638f  (L3, 2026-09-11, sha dc07f1638f73, PR #56160)
TITLE: [Bugfix][MLA] Read sparse model settings from text config (#56160)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse, L3.mla.flashattn_sparse
FILES: vllm/model_executor/layers/attention/sparse_mla_attention.py (+2/-2); vllm/v1/attention/backends/mla/compressor_utils.py (+3/-3); vllm/v1/attention/backends/mla/flashattn_mla_sparse.py (+2/-2); vllm/v1/attention/backends/mla/indexer.py (+4/-4); vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+1/-1); vllm/v1/attention/backends/mla/sparse_utils.py (+1/-1); vllm/v1/attention/backends/mla/xpu_mla_sparse.py (+1/-1); tests/v1/attention/test_dspark_noncausal_sparse_mla.py (+3/-1); tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py (+5/-3); tests/v1/attention/test_sparse_mla_backends.py (+8/-2); (+1 more)
LABELS: bug, rocm, intel-gpu, ready, deepseek, dflash
BODY: ## Purpose ⏎  ⏎ Generic sparse MLA infrastructure reads model-specific settings from the top-level Hugging Face config. Composite and multimodal models keep those settings in their nested text config, which can make backend initialization, workspace sizing, or JIT warmup fail even though the model is configured correctly. ⏎  ⏎ `ModelConfig.hf_text_config` is the normalized accessor for both text-only and composite configurations. ⏎  ⏎ ## Summary ⏎  ⏎ - Read `ind …[truncated]

### L3-0c1e89ceb9  (L3, 2026-09-11, sha 0c1e89ceb92b, PR #55426)
TITLE: [Bugfix][Kimi-K3] Fix KDA projection overlap on Hopper (#55426)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/test_bf16_skinny_gemm.py (+101/-1); vllm/models/kimi_k3/nvidia/low_latency_gemm.py (+47/-28); vllm/models/kimi_k3/nvidia/ops/cute_dsl/kda_skinny_gemm.py (+91/-14)
LABELS: bug, ready, kimi, k3
ISSUES: #55350 [Bug]: [Kimi-K3][Hopper] Low-M TP8 KDA projection fails CUTLASS DSL compilation for sm_90a
BODY: ## Purpose ⏎  ⏎ Fixes #55350. ⏎  ⏎ Kimi-K3's low-token-count TP8 KDA projection-overlap path fails during CUDA ⏎ graph profiling on H100/SM90a. Two independent incompatibilities are involved: ⏎  ⏎ 1. the skinny N/K kernels unconditionally emit PTX `fma.f32.bf16`, which ⏎    requires SM100 or newer and fails libNVVM compilation for `sm_90a`; and ⏎ 2. FlashInfer's `cute-dsl` QKVG backend rejects SM90 during backend validation. ⏎  ⏎ PR #55108 avoids the failur …[truncated]

### L3-cc5dd0a857  (L3, 2026-09-11, sha cc5dd0a857cd, PR #56312)
TITLE: [MRV1] Scope breakable cudagraphs to the piecewise path only (#56312)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/cudagraph/test_breakable_cudagraph.py (+131/-1); vllm/compilation/breakable_cudagraph.py (+19/-12); vllm/compilation/cuda_graph.py (+3/-1); vllm/v1/worker/gpu_model_runner.py (+12/-3)
LABELS: ready, torch.compile, nvidia
BODY: `VLLM_USE_BREAKABLE_CUDAGRAPH=1` no longer overrides FULL cudagraphs in ModelRunnerV1: with `cudagraph_mode=FULL_AND_PIECEWISE`, decode dispatch now uses the standard `CUDAGraphWrapper(FULL)` and breakable capture only replaces `PIECEWISE`, matching ModelRunnerV2's `use_breakable_cg` gating. ⏎  ⏎ `BreakableCUDAGraphWrapper` gains an optional runtime_mode (default keeps match-any behavior for MRV2) and wrapper `unwrap()` recurses through the nested  …[truncated]

### L3-ec6b0494fe  (L3, 2026-09-11, sha ec6b0494fef6, PR #54927)
TITLE: [CI] Bump CUTLASS DSL to 4.7 (#54927)
SOURCES: dependency_pin, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fa4_cutedsl
FILES: cmake/external_projects/tml_fa4.cmake (+1/-1); requirements/cuda.txt (+2/-2)
LABELS: ready, ci/build, nvidia
BODY: ## Summary ⏎  ⏎ - Bump `nvidia-cutlass-dsl[cu13]` from 4.6.2 to 4.7. ⏎ - flashinfer KDA prefill kernel and some future kernels need 4.7 ⏎  ⏎  ⏎ ## Duplicate-work check ⏎  ⏎ Searched open vLLM PRs for `nvidia-cutlass-dsl 4.7`, `CUTLASS DSL 4.7`, `CuTeDSL 4.7`, and the pinned QuACK commit. No open PR implements this upgrade. ⏎  ⏎ The closest results are not duplicates: ⏎  ⏎ - #41834 is a broad DeepSeek V4 SM12x enablement branch and currently uses CUTLASS DSL  …[truncated]

### L3-06e57f622c  (L3, 2026-09-11, sha 06e57f622cf5, PR #56526)
TITLE: [ROCm][Kimi-K3] Fix non-contiguous state_indices crash and GPU-sync assert in fused KDA/MLA prefill (#56526)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+3/-1); vllm/models/kimi_k3/amd/ops/kda_chunk.py (+2/-2)
LABELS: rocm, kimi, k3
BODY: Fixes `evals/gsm8k/test_gsm8k_correctness.py::test_gsm8k_correctness[Kimi-K3-pruned75-DSpark-AITER-TP4]`on ROCm, which failed in two ⏎ stages: ⏎  ⏎  ⏎ - Warm-up crash: fused_kda_chunk asserted state_indices must be a contiguous int32 tensor. The caller passed a non-contiguous column slice of the block table. The wrapper's .to(torch.int32) was a no-op (source was already int32). Fixed by adding an explicit .contiguous() in kimi_k3/amd/ops/kda_chunk.py …[truncated]

### L3-2d75e586fc  (L3, 2026-09-11, sha 2d75e586fcaf, PR #56153)
TITLE: [ROCm][Kernel][DSV4] Remove tl.constexpr to avoid cold-compile churn in indexer gather kernel (#56153)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+8/-8)
LABELS: rocm, ready, DSv4
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ The generic `_cp_gather_indexer_quant_cache_kernel` declared `NUM_TOKENS`, `NUM_BATCHES`, `BLOCK_TABLE_WIDTH`, and `NUM_BLOCKS` as `tl.constexpr`. All four are shape/batch-derived and vary at runtime (total token at each forward pass, sequences per batch, blocks per sequence), so their values entered the Triton compile-cache key and forced a fresh kernel compile for nearly every distinct shape during serving. This problem becomes mo …[truncated]

### L3-9d88ceb026  (L3, 2026-09-11, sha 9d88ceb02694, PR #56485)
TITLE: [KDA] Update flashKDA to support bf16 checkpoint state (#56485)
SOURCES: dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: cmake/external_projects/flashkda.cmake (+1/-1); tests/models/kimi_k3/test_kda.py (+14/-7)
LABELS: ready, ci/build, kimi, k3
BODY: ## Purpose ⏎ Update flashKDA to support bf16 checkpoint state ⏎  ⏎ ## Test Plan ⏎ - tests/models/kimi_k3/test_kda.py ⏎ - Kimi k3 e2e gsm8k ⏎  ⏎ ## Test Result ⏎ ``` ⏎ vllm serve moonshotai/Kimi-K3 \ ⏎   --tensor-parallel-size 8 \ ⏎   --load-format fastsafetensors \ ⏎   --no-enable-flashinfer-autotune \ ⏎   --trust-remote-code \ ⏎   --language-model-only \ ⏎   --attention-config '{"mla_prefill_backend":"TRTLLM_RAGGED","use_prefill_query_quantization":true}' \ ⏎   …[truncated]

### L3-120ec4ebd2  (L3, 2026-09-12, sha 120ec4ebd280, PR #53566)
TITLE: [5/N][warmup][DSv4] Migrate NVIDIA CuTeDSL attention kernels (#53566)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/compressor_utils.py (+1/-2); vllm/v1/attention/backends/mla/indexer.py (+1/-1); vllm/v1/attention/backends/mla/sparse_swa.py (+1/-1); vllm/v1/attention/backends/mla/sparse_utils.py (+1/-1); vllm/v1/attention/ops/common.py (+1/-1); vllm/v1/attention/ops/dcp.py (+1/-1); vllm/v1/worker/block_table.py (+1/-1); tests/model_executor/test_jit_warmup_cutedsl_launcher.py (+38/-3); tests/model_executor/test_jit_warmup_triton_launcher.py (+4/-4); vllm/cute_utils/__init__.py (+8/-0); (+18 more)
LABELS: ready, deepseek, nvidia, DSv4
BODY: Depends on: https://github.com/vllm-project/vllm/pull/50175 ⏎  ⏎ For more details, see parent (draft) PR: https://github.com/vllm-project/vllm/pull/49627 and tracking list issue https://github.com/vllm-project/vllm/issues/49349 ⏎  ⏎ ## Description ⏎  ⏎ This PR migrates the DSv4 NVIDIA CuTeDSL attention and sparse-attention kernels to the shared warmup contract. ⏎  ⏎ ## What Changed ⏎  ⏎ - Migrated indexer-Q and dequantize-and-gather CuTeDSL kernels. ⏎ - Mig …[truncated]

### L3-30118ba27d  (L3, 2026-09-12, sha 30118ba27d1d, PR #56554)
TITLE: [DSV4.1] Remove compressor-aware image sentinel token padding (#56554)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4_1/amd/vl_model.py (+0/-8); vllm/models/deepseek_v4_1/common/mm_preprocess.py (+10/-110); vllm/models/deepseek_v4_1/nvidia/vl_model.py (+0/-8); vllm/transformers_utils/configs/deepseek_v41.py (+0/-6)
LABELS: deepseek, DSv4
BODY: ## Purpose ⏎ - Image sentinel padding only lives in deepseek-v4-vision-exp, not deepseek-v4.1. ⏎ - See: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/blob/main/inference/image_processor.py#L135-L155 and https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/image_processor.py#L140-L173 ⏎  ⏎ ## Test Plan ⏎ TODO ⏎  ⏎ ## Test Result ⏎ TODO ⏎  ⏎ --- ⏎ [details omitted]

### L3-46d2b23ac5  (L3, 2026-09-12, sha 46d2b23ac504, PR #56503)
TITLE: [ROCm][DSV4.1][Perf] Use AITER mHC for the delayed pre block (#56503)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/test_mhc_kernels.py (+95/-0); vllm/_aiter_ops.py (+131/-0); vllm/model_executor/kernels/mhc/__init__.py (+2/-0); vllm/model_executor/kernels/mhc/aiter.py (+90/-0); vllm/model_executor/kernels/mhc/triton.py (+131/-0); vllm/model_executor/layers/mhc.py (+155/-0); vllm/models/deepseek_v4_1/amd/model.py (+16/-19)
LABELS: rocm, ready, deepseek, DSv4
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ DeepSeek-V4.1-Flash on AMD still runs its mHC block as the eager Torch reference (`vllm/models/deepseek_v4_1/amd/model.py` imports `mhc_pre_delayed_torch` / `mhc_post_torch` directly). DeepSeek-V4 was moved onto the `MHCPreOp` / `MHCPostOp` dispatch layer and reaches AITER (#41946, #43950, #52737), but V4.1 was never migrated, so on MI355X it pays ~141 kernel launches per sublayer seam to Sinkhorn-normalize a 4x4 matrix. ⏎  ⏎ That cost do …[truncated]

### L3-d43bb2f37f  (L3, 2026-09-12, sha d43bb2f37f87, PR #53781)
TITLE: [3/N] HiSparse: host-resident sparse-MLA decode hot-buffering (#53781)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.flash_attn.fork_inline_cmake, L3.mla.common_v1, L3.mla.flashmla_sparse, L3.mla.flashinfer_sparse, L3.mla.flashattn_sparse, L3.dispatch.abstract_interface
FILES: CMakeLists.txt (+1/-0); csrc/libtorch_stable/cache_kernels.cu (+70/-5); csrc/libtorch_stable/hisparse_kernels.cu (+1226/-0); csrc/libtorch_stable/ops.h (+52/-1); csrc/libtorch_stable/torch_bindings.cpp (+62/-1); docs/design/hisparse.md (+244/-0); tests/evals/gsm8k/gsm8k_eval.py (+7/-0); tests/kernels/attention/test_mla_cross_layer_kernel_equivalence.py (+8/-4); tests/kernels/test_cp_gather_fp8.py (+42/-0); tests/kernels/test_fused_deepseek_v32_norm_rope.py (+70/-0); (+114 more)
LABELS: documentation, speculative-decoding, ready, torch.compile, ci/build, v1, multi-modality, deepseek, cpu, kv-connector
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Summary ⏎  ⏎ HiSparse adds a host-resident tier beneath vLLM's normal GPU KV cache for sparse-MLA decode. Unlike the original always-host-resident design in #46326, this implementation keeps as much KV as possible in the normal device cache and spills pages to pinned host memory only when GPU capacity must be reclaimed. ⏎  ⏎ Supersedes #46326. ⏎  ⏎ ## Design overview ⏎  ⏎ ```text ⏎ resident GPU pages ⏎         | ⏎         | spill under pressure ⏎         v ⏎   pinned h …[truncated]

### L3-6b153463a8  (L3, 2026-09-12, sha 6b153463a8a6, PR #54007)
TITLE: [Build] Define _USE_MATH_DEFINES for FlashMLA targets (#54007)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.flashmla_build
FILES: cmake/external_projects/flashmla.cmake (+9/-0)
LABELS: ready, ci/build
ISSUES: #53935 [Bug]: M_LOG2E availability depends on build configuration in the FlashMLA extension
BODY: ## Purpose ⏎  ⏎ Fixes #53935. ⏎  ⏎ FlashMLA uses `M_LOG2E` in the vLLM extension (`csrc/extension/sm90/dense_fp8/{softmax.h,pybind.cpp}`) and in four upstream files that vLLM compiles into `_flashmla_C` (`api/dense_decode.h`, `smxx/decode/combine/combine.cu`, `sm90/decode/{dense,sparse_fp8}/splitkv_mla.cuh`). MSVC only defines `M_*` when `_USE_MATH_DEFINES` is set before `<cmath>` is included, so both targets fail to build with NVCC+MSVC. ⏎  ⏎ This adds `_US …[truncated]

### L3-13e221f830  (L3, 2026-09-12, sha 13e221f83084, PR #56562)
TITLE: [Perf] Fuse DSV4.1 input metadata preparation with Triton (#56562)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backend.py (+13/-11); vllm/v1/attention/backends/mla/indexer.py (+69/-52); vllm/v1/attention/ops/metadata.py (+87/-0); tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py (+105/-0)
LABELS: performance, ready, deepseek, DSv4
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ DSV4.1 prepares token-to-request mappings and flattened indexer decode metadata with chains of PyTorch operations on every model step. Replace those operations with Triton kernels that write directly into the existing buffers: ⏎  ⏎ - Build token-to-request mappings from device query boundaries, including zero-length requests and padding. Device boundaries are necessary for DSpark adaptive verification, where CPU boundaries can be stale. ⏎ - …[truncated]

### L3-a0914ab7d0  (L3, 2026-09-12, sha a0914ab7d07b, PR #56398)
TITLE: [Nano-Nemotron] Fix Nano-Nemotron precomputed multimodal embeddings (#56398)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/multimodal/test_nano_nemotron_vl.py (+12/-0); vllm/model_executor/models/nano_nemotron_vl.py (+11/-4)
LABELS: ready, multi-modality
BODY: ## Purpose ⏎  ⏎ Ran into this issue while addressing the mypy fixes for the NO models (#54142). Nano-Nemotron could not correctly consume precomputed multimodal embeddings: ⏎  ⏎ - Tensor-valued `image_embeds` triggered ambiguous tensor truth-value evaluation. ⏎ - `video_embeds` was parsed by the model but never routed through the video path. ⏎ - Parsed video embeddings would be passed to the vision encoder instead of used directly. ⏎  ⏎ This change uses  …[truncated]

### L3-658c8131c7  (L3, 2026-09-12, sha 658c8131c731, PR #54985)
TITLE: [Elastic EP] Reuse CUDA graphs across reconfiguration (#54985)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: requirements/kv_connectors.txt (+1/-1); tests/distributed/test_elastic_ep.py (+33/-65); tests/distributed/test_eplb_utils.py (+11/-4); vllm/config/parallel.py (+17/-1); vllm/distributed/device_communicators/all2all.py (+42/-22); vllm/distributed/device_communicators/base_device_communicator.py (+12/-0); vllm/distributed/elastic_ep/elastic_execute.py (+77/-116); vllm/distributed/elastic_ep/elastic_state.py (+1/-1); vllm/distributed/eplb/eplb_state.py (+66/-12); vllm/engine/arg_utils.py (+6/-0); (+12 more)
LABELS: ci/build, nvidia, mrv2
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Summary ⏎  ⏎ [Async preparation](https://github.com/vllm-project/vllm/pull/47288) and [eager-mode optimizations](https://github.com/vllm-project/vllm/pull/51885) moved most Elastic EP steps out of the blocking commit path, but CUDA graph mode still blocked serving during kernel warmups and CUDA graph capture on every reconfiguration. This PR removes that work from commit as follows: ⏎  ⏎ 1. **Existing ranks reuse their fused-MoE runtime and CUDA g …[truncated]

### L3-29332cf936  (L3, 2026-09-12, sha 29332cf93628, PR #56061)
TITLE: [4/N] Expose HiSparse cache metrics via KV connector stats (#56061)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/libtorch_stable/hisparse_kernels.cu (+19/-1); csrc/libtorch_stable/ops.h (+1/-0); csrc/libtorch_stable/torch_bindings.cpp (+7/-6); tests/v1/kv_connector/unit/test_hisparse_stats.py (+62/-0); tests/v1/worker/test_utils.py (+38/-1); vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py (+37/-0); vllm/distributed/kv_transfer/kv_connector/v1/hisparse/stats.py (+97/-0); vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py (+39/-0); vllm/v1/hisparse/runtime.py (+8/-0)
LABELS: documentation, ready, torch.compile, ci/build, deepseek, kv-connector, nvidia, mrv2, scheduler, kv-cache-manager
BODY: Alternative to #53782, reporting the same HiSparse hot-buffer stats through the existing `KVConnectorStats` pipeline instead of adding a new `finish_step` connector hook. ⏎  ⏎ ## What stays identical ⏎  ⏎ - Kernel counters (`atomicAdd` in the CUDA kernels) and the swap-ordering changes in `runtime.py` are unchanged from the original PR. ⏎ - Same 2000-step async sampling interval with a CUDA event for D2H ordering. ⏎ - Same three Prometheus counters: `vllm:hi …[truncated]

### L3-7bcca16729  (L3, 2026-09-12, sha 7bcca16729b7, PR #56452)
TITLE: [Bugfix] Fix DeepGEMM FP8 warmup coverage (#56452)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/model_executor/test_deep_gemm_warmup.py (+90/-0); vllm/model_executor/kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py (+19/-19); vllm/model_executor/kernels/linear/scaled_mm/deep_gemm.py (+14/-11); vllm/model_executor/warmup/deep_gemm_warmup.py (+17/-69)
LABELS: bug, ready
BODY: # [Bugfix] Fix DeepGEMM FP8 warmup coverage ⏎  ⏎ DeepGEMM warmup missed compressed-tensors and ModelOpt block-FP8 layers, including block-FP8 ParallelLMHead layers. It also excluded supported weight shapes whose output dimension N is divisible by 64 but not 128. The missed compressed-tensors layers caused kernels to JIT compile during requests in the Gemma run below, producing 10–20 second stalls. ⏎  ⏎ The DeepGEMM linear kernel now registers a warmu …[truncated]

### L3-dca96bf97b  (L3, 2026-09-12, sha dca96bf97b35, PR #54416)
TITLE: [Bugfix][Spec Decode] Avoid fastsafetensors deadlock for PP draft models (#54416)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/spec_decode/test_draft_attention_backend_override.py (+8/-1); tests/v1/spec_decode/test_draft_moe_backend_override.py (+8/-1); vllm/v1/worker/gpu/spec_decode/dflash/utils.py (+2/-0); vllm/v1/worker/gpu/spec_decode/dspark/utils.py (+2/-0); vllm/v1/worker/gpu/spec_decode/eagle/utils.py (+4/-0); vllm/v1/worker/gpu/spec_decode/utils.py (+18/-0)
LABELS: bug, speculative-decoding, ready, mrv2, dflash
ISSUES: #50959 [Bug] fastsafetensors ParallelLoader broadcasts on group.WORLD; PP-scoped draft loads deadlock
BODY: ## Summary ⏎  ⏎ - keep `fastsafetensors` enabled for the target model, where every pipeline stage participates in its WORLD-group collectives; ⏎ - fall back to the standard safetensors loader only for an external draft model when PP > 1; ⏎ - apply the PP-safe draft load configuration to EAGLE/EAGLE3, DFlash, and DSpark without changing PP=1 or other load formats. ⏎  ⏎ Fixes #50959. ⏎  ⏎ ## Why a draft-only fallback ⏎  ⏎ With speculative decoding under pipeline paral …[truncated]

### L3-aed894c190  (L3, 2026-09-12, sha aed894c190aa, PR #56323)
TITLE: [6/N][warmup][DSv4] Migrate sampling, and DFlash JIT kernels (#56323)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/models_basic.yaml (+1/-1); .buildkite/test_areas/samplers.yaml (+1/-1); tests/model_executor/test_jit_warmup.py (+32/-0); tests/model_executor/test_jit_warmup_triton_launcher.py (+333/-2); tests/model_executor/test_spec_decode_rejection_warmup.py (+0/-121); tests/v1/sample/test_topk_topp_sampler.py (+4/-0); tests/v1/shutdown/test_delete.py (+13/-3); tests/v1/spec_decode/test_dflash_prepare_inputs.py (+1/-1); tests/v1/worker/test_gpu_autoregressive_speculator.py (+1/-0); vllm/model_executor/layers/mamba/ops/scatter_states.py (+41/-8); (+17 more)
LABELS: speculative-decoding, ready, ci/build, deepseek, cpu, quantization, mrv2, DSv4, dflash
DEEP_STUDY: deep-study: this PR was reverted by PR 56654 (confirmed_revert, reason=unstated)
BODY: Inspired on https://github.com/vllm-project/vllm/pull/56154

### L3-e52be1a62d  (L3, 2026-09-12, sha e52be1a62d38, PR #50178)
TITLE: [9/N][warmup][DSv4] Migrate MHC TileLang kernels (#50178)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/test_mhc_jit_warmup.py (+296/-0); tests/kernels/test_mhc_kernels.py (+11/-2); vllm/model_executor/kernels/mhc/tilelang.py (+117/-264); vllm/model_executor/kernels/mhc/tilelang_kernels.py (+755/-0); vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py (+0/-211); vllm/model_executor/warmup/jit_warmup_tilelang_helper.py (+172/-0); vllm/model_executor/warmup/kernel_warmup.py (+0/-12); vllm/models/deepseek_v4/nvidia/model.py (+59/-0); vllm/models/glm5next/nvidia/model.py (+44/-0)
LABELS: ready, v1, deepseek, DSv4, inkling
BODY: Depends on: https://github.com/vllm-project/vllm/pull/49315 ⏎  ⏎ For more details, see parent (draft) PR: https://github.com/vllm-project/vllm/pull/49627 and tracking list issue https://github.com/vllm-project/vllm/issues/49349 ⏎  ⏎ ## Description ⏎  ⏎ This PR migrates the DSv4 MHC TileLang path to the shared JIT warmup contract. ⏎  ⏎ ## What Changed ⏎  ⏎ - Migrated pre-fusion, post-fusion, fused MHC, head-fusion, and prenorm GEMM TileLang kernels. ⏎ - Move …[truncated]

### L3-e19a3e172e  (L3, 2026-09-12, sha e19a3e172ecc, PR #56629)
TITLE: [5/N] Share HiSparse host cache across TP ranks (reopens #52760) (#56629)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/design/hisparse.md (+10/-0); tests/v1/core/test_kv_cache_utils.py (+24/-2); tests/v1/kv_connector/unit/offloading_connector/test_config.py (+36/-0); tests/v1/kv_offload/cpu/test_shared_offload_region.py (+113/-17); tests/v1/worker/test_attn_utils.py (+82/-0); tests/v1/worker/test_kv_cache_allocation_scope.py (+5/-1); tests/v1/worker/test_utils.py (+253/-0); vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py (+24/-2); vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py (+16/-2); vllm/v1/hisparse/binding.py (+44/-27); (+4 more)
LABELS: documentation, ready, kv-connector, kv-cache-manager
BODY: ## Summary ⏎  ⏎ Share the replicated HiSparse MLA host cache across local TP ranks using one mmap-backed pool. TP rank 0 writes the shared pool; peers wait on its IPC events before reading it. Other executor and parallel layouts retain private pools. ⏎  ⏎ This reopens #52760 against `main`. The original PR was automatically closed when [3/N] (#53781) merged and its base branch, `feat/hisparse-mla-decode-main`, was deleted. ⏎  ⏎ I am not the original author o …[truncated]

### L3-c2ea9f1a5f  (L3, 2026-09-12, sha c2ea9f1a5f10, PR #56478)
TITLE: [Kernel][Perf][Quantization] Fix odd-row performance cliff in per-token-group quantization (#56478)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu (+88/-58); tests/kernels/quantization/test_per_token_group_quant.py (+43/-6)
LABELS: performance, nvidia, quantization, verified
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ Avoid the odd-row performance cliff in per-token-group 8-bit quantization while preserving the existing launch policy for even row counts. ⏎  ⏎ We identified this while investigating unexpectedly poor TPOT (time per output token) with Gemma 4 31B FP8 at TP8 on H100. The reproduction traced about 35 ms of extra time in a 13,359-token prefill forward pass to activation quantization, enough to erase the roughly 30 ms saved by FP8 GEMMs.  …[truncated]

### L3-c711f740b7  (L3, 2026-09-12, sha c711f740b7c7, PR #56349)
TITLE: [ROCm] Auto-enable breakable CUDA graphs for DeepseekV41ForCausalLM (#56349)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/test_config.py (+4/-0); vllm/config/vllm.py (+10/-3)
LABELS: rocm, deepseek, nvidia, DSv4.1
BODY: ## Summary ⏎ - On CUDA, architectures in `DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES` auto-enable `VLLM_USE_BREAKABLE_CUDAGRAPH=1`. On ROCm that helper currently returns an empty set, so DeepSeek-V4.1-Flash dies at capture: `piecewise CUDA graphs (cudagraph_mode=FULL_AND_PIECEWISE) unavailable, model is not torch-compiled and breakable CUDA graph is off`. ⏎ - `DeepseekV41ForCausalLM` does not support `torch.compile`. The ROCm sparse SWA backend only r …[truncated]

### L3-986e2f870a  (L3, 2026-09-12, sha 986e2f870abc, PR #55522)
TITLE: [Refactor][ROCm] Migrate the RDNA3 W4A16 MoE to the oracle/experts pa… (#55522)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/quantization/test_rdna3_compile_guards.py (+68/-59); tests/quantization/test_moe_wna16.py (+34/-0); vllm/config/kernel.py (+2/-0); vllm/model_executor/layers/fused_moe/experts/rdna3_moe.py (+235/-0); vllm/model_executor/layers/fused_moe/oracle/int_wna16.py (+110/-0); vllm/model_executor/layers/fused_moe/routed_experts.py (+0/-1); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py (+0/-7); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16.py (+4/-2); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_rdna3.py (+0/-260); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/rocm_moe_rdna.py (+0/-48)
LABELS: rocm, quantization
BODY: ## Purpose ⏎  ⏎ This is the follow-up refactor @BowenBao asked for when the RDNA3 W4A16 MoE kernel landed in #44075: the kernel is no longer wired in through its own quant-method class, it is now a normal oracle backend with an experts class, like every other WNA16 backend. Tracked in #44460, part of #37753. It was blocked on #44570, which is merged. ⏎  ⏎ In short: ⏎  ⏎ - `WNA16MoEBackend.RDNA3` + `Rdna3WNA16Experts`, selected by the oracle ⏎ - weight p …[truncated]

### L3-72d4d83157  (L3, 2026-09-12, sha 72d4d8315749, PR #56610)
TITLE: [ROCm][Bugfix] Fix elastic EP scaling deadlock (#56610)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/distributed/device_communicators/pynccl.py (+31/-0); vllm/distributed/elastic_ep/elastic_execute.py (+13/-2); vllm/distributed/elastic_ep/standby_state.py (+23/-14); vllm/distributed/parallel_state.py (+39/-22)
LABELS: bug, rocm, ready
BODY: ## Purpose ⏎  ⏎ Elastic EP scaling sometimes hangs during a scale up/down on ROCm. This was seen from time to time in CI, e.g. [here](https://buildkite.com/vllm/amd-ci/builds/12870/list?sid=01a08fb2-f7cc-4c10-b517-929727fc69e1&tab=output). ⏎  ⏎ This was root-caused via RCCL logs: during elastic-EP prepare the standby/joining groups are built while the engine is still serving, so a new communicator warm-up (which does an all-reduce) might run concurre …[truncated]

### L3-ebe1dec2da  (L3, 2026-09-12, sha ebe1dec2da16, PR #56157)
TITLE: [PCP][DCP] Enable PCP+DCP on sparse-MLA models (#56157)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashmla_sparse, L3.dispatch.abstract_interface
FILES: vllm/model_executor/layers/attention/mla_attention.py (+7/-0); vllm/model_executor/layers/attention/sparse_mla_attention.py (+8/-3); vllm/v1/attention/backend.py (+15/-2); vllm/v1/attention/backends/mla/flashmla_sparse.py (+215/-57); vllm/v1/attention/backends/mla/indexer.py (+210/-20); vllm/v1/attention/backends/mla/sparse_utils.py (+52/-7); vllm/v1/worker/gpu/attn_utils.py (+8/-0); vllm/v1/worker/gpu/input_batch.py (+3/-0); vllm/v1/worker/gpu/model_states/default.py (+2/-0); vllm/v1/worker/gpu/pcp_manager.py (+70/-26); (+11 more)
LABELS: ready, ci/build, deepseek, nvidia, mrv2
ISSUES: #53573 [Bug]: [PCP+DCP][MLA] Rank-local PCP context metadata causes divergent DCP KV-gather collectives
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Enable DCP to run alongside PCP on sparse-MLA models, so a deployment can shard the KV cache for decode and still split prefill query work across ranks. ⏎  ⏎ This change: ⏎  ⏎ - gives the sparse indexer a DCP-gathered prefill path: this rank's KV shard is all-gathered across the DCP group ⏎ - gives the FlashMLA-sparse backend the matching prefill path, with a rank-major gathered KV workspace and a PCP-invariant chunk plan every rank deri …[truncated]

### L3-fa008bdccf  (L3, 2026-09-13, sha fa008bdccf10, PR #56464)
TITLE: [Perf][Kernel] Integrate DeepSelect TopK for the DSA sparse indexer (#56464)
SOURCES: dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-0); cmake/external_projects/deepselect.cmake (+97/-0); setup.py (+4/-0); tests/kernels/test_top_k_per_row.py (+323/-0); vllm/config/kernel.py (+32/-0); vllm/engine/arg_utils.py (+16/-1); vllm/model_executor/layers/indexer_topk.py (+339/-0); vllm/model_executor/layers/sparse_attn_indexer.py (+17/-51)
LABELS: performance, ready, ci/build
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## What ⏎  ⏎ Integrate [DeepSeek DeepSelect](https://github.com/deepseek-ai/DeepSelect) (MIT) — a high-performance TopK kernel library for DeepSeek Sparse Attention — into vLLM via CMake FetchContent, and make **every** decode top-k implementation on the DSA sparse indexer path explicitly selectable, with a measured auto heuristic on top. ⏎  ⏎ - New `vllm/model_executor/layers/indexer_topk.py`: the indexer's decode top-k stage. Hosts every backend en …[truncated]

### L3-e7a3963339  (L3, 2026-09-13, sha e7a396333965, PR #56682)
TITLE: [Bugfix] Avoid repeated dummy initialization and random CPU Engram fills (#56682)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/model_loader/dummy_loader.py (+6/-2); vllm/model_executor/model_loader/weight_utils.py (+4/-0); vllm/models/deepseek_v4_1/common/engram.py (+4/-0)
LABELS: bug, ready, deepseek, DSv4.1
BODY: ## Purpose ⏎  ⏎ Avoid repeatedly initializing descendant weights under `--load-format dummy`. The loader already walks `model.modules()`, so each iteration now initializes only local parameters and persistent buffers. This matches default PyTorch state selection without recursively revisiting descendants and preserves parameter attributes used for dummy initialization. ⏎  ⏎ CPU-offloaded DeepSeek V4.1 Engram tables request constant dummy weights of 1.0 a …[truncated]

### L3-d2d649e674  (L3, 2026-09-13, sha d2d649e674c7, PR #56512)
TITLE: [DS V4.1][Engram] Support async prefetch for offloaded engram lookups and engram DP sharding (#56512)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: .buildkite/test_areas/distributed.yaml (+4/-0); tests/distributed/test_engram_dp_shard.py (+776/-0); tests/engine/test_arg_utils.py (+22/-8); tests/kernels/test_engram.py (+226/-14); tests/test_config.py (+110/-48); vllm/config/engram.py (+45/-15); vllm/config/vllm.py (+23/-8); vllm/distributed/parallel_state.py (+80/-0); vllm/envs.py (+3/-6); vllm/models/deepseek_v4_1/common/engram.py (+112/-93); (+2 more)
LABELS: ready, ci/build, qwen, deepseek, DSv4.1
BODY: Re-open of #56357, rebased onto `main`. The original PR was stacked on ⏎ the `dsv41-feat` staging branch and was auto-closed when that branch was ⏎ removed after DeepSeek V4.1 merged to main. It now targets `main` ⏎ directly and also carries the engram DP-shard support commit that was ⏎ previously split out as #56230 (also auto-closed for the same reason). ⏎  ⏎ ## What's New ⏎  ⏎ Three commits: engram DP sharding (from #56230) plus two engram ⏎ embedding  …[truncated]

### L3-2671fedfc7  (L3, 2026-09-13, sha 2671fedfc7ae, PR #56654)
TITLE: [MRV2] Revert explicit Triton JIT warmup migration (#56654)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/spec_decode/test_dflash_prepare_inputs.py (+1/-1); tests/v1/worker/test_gpu_autoregressive_speculator.py (+0/-1); vllm/v1/worker/gpu/model_runner.py (+44/-53); vllm/v1/worker/gpu/sample/states.py (+1/-5); vllm/v1/worker/gpu/spec_decode/dflash/speculator.py (+39/-92); vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py (+12/-76)
LABELS: speculative-decoding, ready, mrv2, dflash
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 56323 reason=unstated
BODY: ## Purpose ⏎  ⏎ Revert the MRV2-specific Triton JIT warmup migration introduced by #56323. Restore direct DFlash and multi-module MTP kernel launches, remove their warmup providers and MRV2 sampler registration, and restore the previous sampler initialization and dummy-run behavior. The two associated test adjustments are also reverted. ⏎  ⏎ Shared model/attention JIT warmup infrastructure and normal dummy-run/CUDA graph warmup remain in place. The six f …[truncated]

### L3-0cf266a469  (L3, 2026-09-13, sha 0cf266a469bd, PR #56639)
TITLE: [Pooling] Support prompt embeddings in MRV2 decoder pooling (#56639)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/model_runner.py (+0/-1); vllm/v1/worker/gpu/pool/pooling_runner.py (+2/-1)
LABELS: ready, mrv2
BODY: ## Purpose ⏎  ⏎ A regression bug fix between MRV1 and MRV2 for pooling decoder models (ie. `Qwen/Qwen3-Embedding-0.6B`, etc), ⏎  ⏎ MRV2's prompt-embedding state supports precomputed embeddings, but the pooling request path unconditionally asserted that `prompt_token_ids` was present. Decoder pooling requests that supplied `prompt_embeds` therefore crashed before inference, even though most pooling tasks do not need token IDs. ⏎  ⏎ This PR makes the poo …[truncated]

### L3-dd4c841070  (L3, 2026-09-13, sha dd4c8410707b, PR #55897)
TITLE: [LoRA] Add LoRA support for DeepSeek-V4 Flash Vision (#55897)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/models/supported_models.md (+1/-1); vllm/lora/ops/triton_ops/lora_shrink_op.py (+1/-1); vllm/models/deepseek_v4/common/vl_model.py (+40/-1)
LABELS: documentation, deepseek, DSv4
ISSUES: #55683 [Feature]: LoRA support for deepseek v4 flash vision
BODY: Add LoRA support for DeepSeek-V4-Flash-Vision-Exp. ⏎  ⏎ Fixes #55683. ⏎  ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-a987777755  (L3, 2026-09-13, sha a987777755c8, PR #56652)
TITLE: [Bugfix][Gemma4] Keep image kwargs out of video preprocessing (#56652)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/multimodal/processing/test_gemma4.py (+68/-0); vllm/model_executor/models/gemma4_mm.py (+3/-0)
LABELS: bug, ready, multi-modality
BODY: ## Purpose ⏎  ⏎ Gemma4 processes video frames through the HF image processor, but currently forwards image-only overrides to that call. For example, a mixed image/video request with: ⏎  ⏎ ```python ⏎ mm_processor_kwargs={"images_kwargs": {"max_soft_tokens": 560}} ⏎ ``` ⏎  ⏎ fails because the video path also passes `max_soft_tokens=70` as a top-level argument: ⏎  ⏎ ```text ⏎ ValueError: Keyword argument max_soft_tokens was passed two times: ⏎ in a dictionary for images_k …[truncated]

### L3-8cf9de9080  (L3, 2026-09-13, sha 8cf9de9080ea, PR #56528)
TITLE: [Test][Determinism] Cover VLM batch invariance in default execution mode (#56528)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/determinism/test_batch_invariance_vlm.py (+41/-7)
LABELS: ready
BODY: ## Summary ⏎  ⏎ - add a focused Qwen3-VL batch-invariance case for the default (non-eager) execution mode ⏎ - keep the existing eager image/video coverage while sharing the common runner ⏎ - fail closed when selected-token logprobs or the expected number of outputs/tokens are missing ⏎ - disable prefix/MM processor caches and the FlashInfer sampler so the test isolates the intended execution-mode comparison ⏎  ⏎ ## Why this is not duplicate work ⏎  ⏎ #53531 added  …[truncated]

### L3-71888f507a  (L3, 2026-09-13, sha 71888f507ad5, PR #56688)
TITLE: [Perf] Avoid redundant conversions in FP8 dummy initialization (#56688)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/model_loader/weight_utils.py (+3/-4)
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Avoid redundant conversions when initializing FP8 dummy weights. Allocate the FP16 scratch tensor with `empty_like`, fill it, and let `param.copy_` convert directly into the destination. This avoids reading old parameter values that are immediately overwritten and removes the intermediate FP8 allocation/copy. ⏎  ⏎ For a 128 Mi-element FP8 tensor on one GB200, median GPU time over 10 iterations decreased from 0.665 ms to 0.301 ms; peak scr …[truncated]

### L3-b23433088b  (L3, 2026-09-13, sha b23433088bf2, PR #55802)
TITLE: [Attention] Remove DCP indexer interleave guard and test TP1 output parity (#55802)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+0/-9)
LABELS: ready, ci/build, nvidia, glm
BODY: ## Purpose ⏎  ⏎ Remove the sparse indexer's rejection of DCP with `cp_kv_cache_interleave_size > 1`, and add direct attention-output regression tests against complete, unsharded TP1 KV. This enables review of configurations such as DCP with NIXL, which uses interleave 64. ⏎  ⏎ **Draft: the tested attention outputs agree within the stated FP8 tolerances. Full-model outputs are numerically sensitive, and accuracy equivalence is not established.** The origi …[truncated]

### L3-82a85dc1d2  (L3, 2026-09-13, sha 82a85dc1d2d5, PR #56305)
TITLE: [Attention] Add Triton/FlashInfer composite for multimodal prefix attention (#56305)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.triton.v1_backend, L3.triton.triton_flashinfer, L3.dispatch.selector, L3.dispatch.registry, L3.dispatch.abstract_interface, L3.platform.cuda_selection, L3.flex_attention
FILES: vllm/platforms/cuda.py (+8/-0); vllm/v1/attention/backend.py (+10/-0); vllm/v1/attention/backends/composite.py (+509/-0); vllm/v1/attention/backends/flash_attn.py (+4/-0); vllm/v1/attention/backends/flex_attention.py (+4/-0); vllm/v1/attention/backends/registry.py (+6/-0); vllm/v1/attention/backends/triton_attn.py (+4/-0); vllm/v1/attention/backends/triton_flash_attn.py (+67/-0); vllm/v1/attention/backends/triton_flashinfer.py (+22/-0); vllm/v1/attention/backends/utils.py (+19/-2); (+5 more)
LABELS: documentation, speculative-decoding, ready, nvidia, dflash
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: > **TTFT:** Without `--request-rate`, faster decode increases offered load. At fixed 5 req/s, FP8 text TTFT is **34.5% lower**. ⏎  ⏎ Triton image masks + FlashInfer causal attention; reusable composite, no worker/core changes. ⏎  ⏎ Alternatives: #48162 phase routing; #47118 model-specific/FP8; #46558 FlashInfer custom masks. ⏎  ⏎ ## End-to-end benchmarks ⏎  ⏎ B300 ×4 TP, `google/gemma-4-31B-it`, BF16 weights; BF16 or FP8 E4M3 KV, `--max-num-seqs 128`. ⏎ Baseline:  …[truncated]

### L3-586f652f8d  (L3, 2026-09-13, sha 586f652f8d9e, PR #56715)
TITLE: [Bugfix][PCP][DCP] Respect interleave in indexer KV gather mapping (#56715)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+8/-2); tests/v1/attention/test_indexer_dcp_localize.py (+32/-11)
LABELS: bug, ready
BODY: PCP’s index gather assumes interleave 1. With DCP4/interleave64, token 1 reads the key for token 64. Use the configured interleave when mapping gathered shards back to token positions, and keep padded gathers in bounds. No new kernel. ⏎  ⏎ Separate from #55802, which removes the interleave restriction; this PR leaves that guard unchanged. Duplicate searches found no open PR fixing this PCP gather mapping. ⏎  ⏎ Validation on main `b6e2aa748b`: **72 orderi …[truncated]

### L3-186a1e6222  (L3, 2026-09-13, sha 186a1e62221c, PR #55768)
TITLE: [Warmup] Gemma 4 de-JITification (#55768)
SOURCES: path_core, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fa4_cutedsl
FILES: vllm/v1/attention/backends/flash_attn.py (+227/-10); vllm/vllm_flash_attn/flash_attn_interface.py (+44/-10); vllm/v1/worker/gpu/attn_utils.py (+4/-1); vllm/v1/worker/gpu/model_runner.py (+12/-6)
LABELS: speculative-decoding, ready, quantization, mrv2, dflash
BODY: 

### L3-f09c52a587  (L3, 2026-09-13, sha f09c52a587f2, PR #56170)
TITLE: [ROCm][Performance] Avoid blocking MiniMax M3 scalar upload (#56170)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/models/minimax_m3/common/sparse_attention.py (+4/-1)
LABELS: rocm, ready, minimax
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ MiniMax M3 initializes the first prefill cumulative KV length with a Python ⏎ scalar assignment. On ROCm, PyTorch lowers that assignment to a four-byte ⏎ `hipMemcpyWithStream`. If earlier work is queued on the compute stream, the API ⏎ waits for that work before returning to the host, delaying subsequent command ⏎ submission. ⏎  ⏎ Use an in-place device zero on a one-element view for ROCm. This enqueues the ⏎ initialization and preserves s …[truncated]

### L3-dec0b5d63f  (L3, 2026-09-13, sha dec0b5d63fc4, PR #56676)
TITLE: [Bugfix] Skip Triton autotune inspection without Triton (#56676)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/model_executor/test_jit_warmup_triton_launcher.py (+20/-0); vllm/model_executor/warmup/jit_warmup_triton_helper.py (+3/-1)
LABELS: bug, ready
BODY: ## Summary ⏎  ⏎ Nightly main build #88612 exposed deterministic import failures in Arm CPU test shards 2 and 3. The CPU image correctly disables Triton and installs `TritonPlaceholder`, but importing `vllm.v1.spec_decode.utils` constructs a warmup dispatcher whose autotune check unconditionally dereferences `triton.runtime`. ⏎  ⏎ This change returns `False` before Triton autotune inspection when `HAS_TRITON` is false. Runtime dispatch behavior is unchang …[truncated]

### L3-3b533197a9  (L3, 2026-09-13, sha 3b533197a941, PR #56683)
TITLE: [Perf] Parallelize mHC pre-norm JIT warmup (#56683)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/kernels/mhc/warmup.py (+16/-0); vllm/model_executor/warmup/jit_warmup.py (+7/-4)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Compile independent mHC pre-norm warmup variants concurrently. Add a `compile_many()` hook to `VllmJitKernel`, retain sequential compilation by default, and use TileLang's `par_compile()` for missing `MHCPreNormKernel` variants with at most four compiler workers per process. Cached variants are retained and duplicate keys are removed. TileLang elaborates the modules serially before compiling them in parallel. ⏎  ⏎ The existing open PRs #5 …[truncated]

### L3-9f03b510c3  (L3, 2026-09-13, sha 9f03b510c335, PR #55914)
TITLE: [DSv4 Bug] fix dsv4 start up error `NotImplementedError: DeepSeek V4 MegaMoE currently requires expert parallel` (#55914)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/config/speculative.py (+1/-0)
LABELS: bug, ready, deepseek, DSv4
BODY: ## Purpose ⏎  ⏎ `vllm serve deepseek-ai/DeepSeek-V4-Pro-0813   --trust-remote-code   --kv-cache-dtype fp8   --block-size 256   --enable-expert-parallel   --tensor-parallel-size 8   --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE", "custom_ops":["all"]}'   --attention_config.use_fp4_indexer_cache=True  --moe-backend deep_gemm_mega_moe   --tokenizer-mode deepseek_v4   --tool-call-parser deepseek_v4   --enable-auto-tool-choice   --reasonin …[truncated]

### L3-7fe8fc803d  (L3, 2026-09-13, sha 7fe8fc803d90, PR #56645)
TITLE: [NIXL][PCP][DCP] Expose PCP producer KV shards as transfer ranks (#56645)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_nixl_connector.py (+57/-16); tests/v1/kv_connector/unit/test_nixl_push_connector.py (+9/-6); tests/v1/kv_connector/unit/test_tp_mapping.py (+1/-0); tests/v1/kv_connector/unit/utils.py (+2/-0); vllm/config/vllm.py (+6/-6); vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py (+6/-0); vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py (+19/-9); vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py (+10/-3); vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py (+1/-1); vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py (+1/-1); (+3 more)
LABELS: ready, deepseek, kv-connector, mrv2
BODY: Expose TP1 PCP+DCP KV shards as NIXL transfer ranks. Preserve completion reports from every sharded PCP rank in `get_transfer_results()`; replicated PCP still reports only from rank zero. ⏎  ⏎ Builds on merged #56157; replaces #54496. Distinct from #45340’s block accounting. ⏎  ⏎ Validation: **18 PCP producer/topology tests passed on B300 with standard fixtures, including NCCL initialization. 72 TP-mapping/push tests passed on CPU.** Before the guard, a  …[truncated]

### L3-214248c7d8  (L3, 2026-09-13, sha 214248c7d896, PR #56695)
TITLE: [CI] Move smaller H200 workloads to 18GB MIG slices (#56695)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/cuda.yaml (+2/-2); .buildkite/test_areas/engine.yaml (+17/-3); .buildkite/test_areas/misc.yaml (+2/-2); .buildkite/test_areas/samplers.yaml (+19/-4)
LABELS: ci/build, nvidia
BODY: Move CUDAGraph and V1 Executor + Worker to `h200_18gb`. Split Samplers and E2E Core so their smaller tests use 18GB slices while these cases stay on `h200_35gb`: ⏎ - Qwen2-Audio multimodal beam search, with both sampler modes. ⏎ - All six Gemma-3n KV-sharing fast-prefill variants. ⏎ - Both Mamba prefix-cache MRV2 variants (synchronous and asynchronous). ⏎  ⏎ Existing AMD test coverage is preserved. No model, context-length, or memory-budget settings are ch …[truncated]

### L3-dfa1984e58  (L3, 2026-09-13, sha dfa1984e58dc, PR #55235)
TITLE: [ROCm][Perf] Tune MiniMax-M3 decode top-k for short contexts (#55235)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_minimax_m3.py (+63/-41); vllm/models/minimax_m3/amd/ops/index_topk.py (+27/-7)
LABELS: rocm, ready, minimax
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ - Use one B128/C1/W2/S1 CTA for the gfx950 MiniMax-M3 decode selector when `topk == 16` and the graph capacity is at most 128 sparse blocks. ⏎ - Keep the existing launch policy unchanged for larger capacities, other top-k values, and other architectures. ⏎ - Extend the selector tests to cover the production 74-block capacity, the 128/129 dispatch boundary, exact ordering, and graph replay. ⏎  ⏎ ## Kernel performance ⏎  ⏎ Measured on gfx950 at the …[truncated]

### L3-73d2a8cf86  (L3, 2026-09-13, sha 73d2a8cf86c4, PR #51700)
TITLE: [2/2][Model Runner V2] FULL CUDA graph capture for microbatched steps (DBO) (#51700)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/worker/test_gpu_ubatch_slicing.py (+159/-4); vllm/config/vllm.py (+3/-2); vllm/v1/worker/gpu/cudagraph_utils.py (+116/-30); vllm/v1/worker/gpu/dp_utils.py (+34/-12); vllm/v1/worker/gpu/model_runner.py (+6/-1); vllm/v1/worker/gpu/ubatch_utils.py (+89/-31)
LABELS: ready, nvidia, mrv2
BODY: DBO for Model Runner V2 (RFC: #50738 ) is two PRs: ⏎  ⏎ #50945 [1/2][Model Runner V2] DBO support, eager mode (P0–P2)  ⏎ -> #51700 [2/2][Model Runner V2] FULL CUDA graph capture for microbatched steps (P3–P4) ⏎ DBO for Model Runner V2 ([RFC #50738](https://github.com/vllm-project/vllm/issues/50738)) is two PRs: ⏎  ⏎ Stacked on top of #50945; the diff below contains that PR's commits too, so review only the last three (`[Feat][WIP] FULL cudagraph captur …[truncated]

### L3-cf1584f373  (L3, 2026-09-13, sha cf1584f373d0, PR #56741)
TITLE: [Refactor] Normalize DeepSeek V4.1 model package naming (#56741)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.registry
FILES: vllm/models/deepseek_v41/nvidia/flashmla.py (+3/-3); .buildkite/test_areas/distributed.yaml (+1/-1); .github/mergify.yml (+2/-3); tests/distributed/test_engram_dp_shard.py (+5/-5); tests/kernels/core/test_fused_q_kv_rmsnorm.py (+4/-4); tests/kernels/test_compressor_kv_cache.py (+5/-5); tests/kernels/test_engram.py (+6/-6); tests/kernels/test_fused_indexer_q_rope_quant.py (+3/-3); tests/kernels/test_mhc_kernels.py (+3/-3); tests/models/test_deepseek_v4_mega_moe.py (+2/-2); (+34 more)
LABELS: new-model, ready, ci/build, deepseek, nvidia, quantization, dflash, DSv4.1
BODY: ## Purpose ⏎  ⏎ Rename `vllm/models/deepseek_v4_1/` to `vllm/models/deepseek_v41/`, following the existing `deepseek_v32` convention and the V4.1 config/frontend naming. Update Python imports, lazy model and attention registration strings, test references, Buildkite path filters, and auto-label rules together. ⏎  ⏎ Searched open upstream PRs for DeepSeek naming changes and rename titles; found no overlapping package migration. Existing model class names, …[truncated]

### L3-a2685f2cda  (L3, 2026-09-14, sha a2685f2cdace, PR #54323)
TITLE: [Bugfix][Multimodal] Validate base64 video payloads, matching image and audio (#54323)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/multimodal/media/video.py (+1/-1)
LABELS: bug, ready, multi-modality
BODY: > **Correction (updated after re-checking the default backend).** The original ⏎ > title and body of this PR claimed the malformed-video path returns HTTP 500. ⏎ > That is wrong for the default configuration, and I have retitled and rewritten ⏎ > accordingly. Under `VLLM_VIDEO_LOADER_BACKEND=opencv` (the default), ⏎ > `open_video_capture` raises `ValueError("Could not open video stream")`, which ⏎ > already maps to 400. I verified this by reading ⏎ > `vllm/m …[truncated]

### L3-78e84261ab  (L3, 2026-09-14, sha 78e84261ab2f, PR #48498)
TITLE: [Performance] Add Triton kernel for Gemma3n sparse GELU (#48498)
SOURCES: release_notes
ARTIFACT_HINTS: L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: benchmarks/kernels/benchmark_gelu_and_mul_sparse.py (+90/-0); tests/compile/passes/ir/test_lowering.py (+34/-0); tests/compile/test_aot_compile.py (+65/-0); tests/kernels/ir/test_activation.py (+183/-0); tests/test_config.py (+16/-0); vllm/config/kernel.py (+3/-0); vllm/ir/ops/__init__.py (+2/-1); vllm/ir/ops/activation.py (+34/-0); vllm/kernels/__init__.py (+2/-2); vllm/kernels/triton/__init__.py (+4/-0); (+5 more)
LABELS: performance, rocm, intel-gpu, ready, torch.compile, nvidia, vllm-ir
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ This PR adds a dedicated Triton provider for Gemma3n's `gelu_and_mul_sparse` activation. The kernel fuses per-row mean and variance, Gaussian thresholding, tanh GELU, and gated multiplication, avoiding the multi-kernel native/Inductor path. ⏎  ⏎ It also: ⏎  ⏎ - registers `gelu_and_mul_sparse` with vLLM IR and routes `GeluAndMulSparse` through IR dispatch; ⏎ - selects `triton` before `native` on CUDA when the dtype, layout, approximation mode, a …[truncated]

### L3-b443c1cc4e  (L3, 2026-09-14, sha b443c1cc4e12, PR #55737)
TITLE: [Perf][GLM-5.3-Flash] Use FlashKDA for KDA chunked prefill (1.7-3.8x faster than the Triton chunk path) (#55737)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/glm5next/nvidia/kda.py (+144/-28)
LABELS: ready, glm
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ One of three independent GLM-5.3-Flash perf PRs  ⏎  ⏎ Use **FlashKDA** (`vllm._flashkda_C`, already built and used by Kimi-K3) for GLM-5.3-Flash's KDA chunked prefill instead of the ~15-kernel Triton `chunk_kda_with_fused_gate` path. ⏎  ⏎ FlashKDA implements the same bounded-gate KDA recurrence GLM-5.3-Flash uses (`lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`, in-kernel q/k l2norm, raw beta logits), so it is a drop-in. On GB300 pe …[truncated]

### L3-238cb2b191  (L3, 2026-09-14, sha 238cb2b19163, PR #55738)
TITLE: [Perf][GLM-5.3-Flash] Dense/masked-MHA sparse prefill for the NoPE (256, 0, 256) layout + skip the NoPE K concat (#55738)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+4/-0); vllm/model_executor/layers/attention/sparse_mla_attention.py (+2/-0); vllm/v1/attention/backends/mla/prefill/flash_attn.py (+7/-0); tests/v1/attention/test_mla_backends.py (+20/-0); tests/v1/attention/test_mla_prefill_selector.py (+34/-0); tests/v1/attention/test_sparse_mla_backends.py (+29/-0)
LABELS: ready, needs-rebase, glm
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ One of three independent GLM-5.3-Flash perf PRs  ⏎  ⏎ On main, GLM-5.3-Flash's MLA layout (qk_nope 256, qk_rope 0, v 256) is unknown to the FlashAttention prefill backend and to the masked-MHA allow-list, so `MLACommonImpl` logs `No MLA prefill backend supports this model` and **every prefill token goes through the per-token top-k MQA kernel**. Profiling shows that kernel is KV-gather bound at the HBM roofline: 5.9 ms/layer per 16k-token  …[truncated]

### L3-c676e4930b  (L3, 2026-09-14, sha c676e4930bb3, PR #55133)
TITLE: [Spec Decode] Fix Qwen3 DSpark d2t requirement for padded-vocab drafts (#55133)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/model_executor/test_qwen3_omni.py (+33/-13); vllm/model_executor/models/qwen3_dspark.py (+5/-2)
LABELS: ready, qwen, dflash
BODY: ## Purpose ⏎  ⏎ `Qwen3DSparkForCausalLM` flags *any* draft whose vocab size differs from the target as reduced-vocabulary, requiring an `lm_head` + `d2t` mapping: ⏎  ⏎ ```python ⏎ uses_reduced_vocab = self.config.draft_vocab_size != self.target_vocab_size ⏎ ``` ⏎  ⏎ But a draft can be **larger** than the target's *logical* `vocab_size` when padded to its *physical* embedding size — e.g. `Inkling-Small-NVFP4`: `get_vocab_size()` = 200058, but `embed`/`unembed` ar …[truncated]

### L3-ff5f6d41b1  (L3, 2026-09-14, sha ff5f6d41b1cc, PR #50894)
TITLE: [Bugfix] Scale KV page size for hidden states extraction with TP (#50894)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/core/test_kv_cache_utils.py (+39/-0); vllm/v1/core/kv_cache_utils.py (+34/-0)
LABELS: bug, ready, kv-cache-manager
ISSUES: #51016 [Bug]: extract_hidden_states fails with TP>1 when KV page size < hidden-state per-token cost
BODY: ## Summary ⏎ - When `extract_hidden_states` is combined with tensor parallelism, the target model's KV page size shrinks (`num_kv_heads / TP`) but the hidden-state per-token cost stays at full `hidden_size`. This causes an assertion failure in `KVCacheSpecBase.__post_init__`: `assert self.page_size_padded >= real_page_size` ⏎ - Scale up all target group block sizes when the hidden-state per-token cost exceeds the common page, before aligning hidden-s …[truncated]

### L3-4be3dcf0fc  (L3, 2026-09-14, sha 4be3dcf0fc7a, PR #56722)
TITLE: [PCP][DCP] Declare FlashMLASparse MTP support at CP interleave > 1 (#56722)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.flashmla_sparse
FILES: vllm/v1/attention/backends/mla/flashmla_sparse.py (+1/-0)
BODY: ## Purpose ⏎  ⏎ Make MTP reachable for PCP+DCP sparse-MLA prefillers behind NIXL P/D. ⏎  ⏎ With `decode_context_parallel_size > 1` and a `NixlConnector`, `VllmConfig.adjust_dcp_kv_cache_interleave_size` raises `cp_kv_cache_interleave_size` to the block size (64) for block-level alignment of the DCP → decoder transfer. `cp_utils.validate_cp_support` then asserts `supports_mtp_with_cp_non_trivial_interleave_size` on every attention impl whenever a spec …[truncated]

### L3-03dc26e639  (L3, 2026-09-14, sha 03dc26e639e1, PR #53793)
TITLE: [Perf][Kernel][Quantization] Fuse ReLU2 with static FP8 activation quantization (#53793)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/compile/passes/test_silu_mul_quant_manual_fusion.py (+6/-5); tests/fusion/test_quant_activation_contract.py (+3/-2); tests/kernels/test_fused_quant_activation.py (+26/-4); tests/kernels/test_relu2_fp8_quant.py (+229/-0); tests/lora/test_layers.py (+10/-0); tests/model_executor/test_nemotron_h_quantization.py (+27/-0); vllm/lora/layers/base_linear.py (+4/-0); vllm/model_executor/layers/fusion/fused_act_quant.py (+54/-12); vllm/model_executor/layers/fusion/quant_activation.py (+15/-7); vllm/model_executor/layers/fusion/relu2_fp8_quant.py (+76/-0); (+2 more)
LABELS: ready, torch.compile, llama, nvidia, quantization
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Fuse BF16 ReLU2 with static per-tensor FP8 activation quantization for ⏎ compatible Nemotron-H MLP down-projections. The Triton producer preserves the ⏎ observable BF16 rounding boundary and returns a `QuantizedActivation` through ⏎ the generic `maybe_fused_act_quant` interface, allowing the consumer to use its ⏎ checkpoint input scale without quantizing twice. ⏎  ⏎ This implements the activation-quantization work tracked by #43501, integrates ⏎ wi …[truncated]

### L3-eb42686a30  (L3, 2026-09-14, sha eb42686a30cd, PR #56299)
TITLE: [Bugfix][Frontend] Support Responses text types in DeepSeek V4.1 (#56299)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/tokenizers_/test_deepseek_v41.py (+18/-0); vllm/tokenizers/deepseek_v41.py (+1/-1)
LABELS: bug, ready, deepseek, DSv4.1
ISSUES: #56297 [Bug]: DeepSeek V4.1 rejects Responses API text content types
BODY: ## Purpose ⏎  ⏎ Fixes #56297. ⏎  ⏎ Normalize Responses API `input_text` and `output_text` parts as text in the DeepSeek V4.1 tokenizer. Existing image handling is unchanged. ⏎  ⏎ No matching issue or PR was open when this work started. ⏎  ⏎ ## Test Plan ⏎  ⏎ ```bash ⏎ VLLM_TARGET_DEVICE=cpu .venv/bin/python -m pytest tests/tokenizers_/test_deepseek_v41.py -q ⏎ .venv/bin/pre-commit run --files vllm/tokenizers/deepseek_v41.py tests/tokenizers_/test_deepseek_v41.py ⏎ ``` ⏎  ⏎ ## …[truncated]

### L3-a6c5d6d0fc  (L3, 2026-09-14, sha a6c5d6d0fcd7, PR #51794)
TITLE: [ROCm][Perf] Enable CSA multi-stream overlap for DeepSeek-V4 (#51794)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/amd/model.py (+1/-10); vllm/models/deepseek_v4/amd/rocm.py (+194/-0); vllm/models/deepseek_v4/attention.py (+17/-14); vllm/models/deepseek_v4/cpu/cpu_sparse.py (+2/-1)
LABELS: rocm, ready, deepseek, cpu, nvidia, DSv4
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Enable kernel-level multi-stream overlap for DeepSeek-V4 CSA (Compressed Sparse Attention, compress_ratio=4) attention layers on ROCm. When enabled, the attention forward forks three HIP streams before the input GEMMs: ⏎  ⏎ - **Default stream:** fused wqa+wkv GEMM → q/kv RMSNorm → wq_b → qnorm/RoPE → SWA KV insert. ⏎ - **Aux stream 0:** main-compressor wkv_gate GEMM → compress kernels (compressed KV cache write). ⏎ - **Aux stream 1:** i …[truncated]

### L3-6623fe5b4e  (L3, 2026-09-14, sha 6623fe5b4ec7, PR #49417)
TITLE: [Bugfix] MiniCPM-V 4.6: fix ViT self-attn qkv weight loading (#49417)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/minicpmv4_6.py (+6/-13)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ `openbmb/MiniCPM-V-4.6` currently **fails to load under vLLM** — engine init aborts, so the model cannot be served at all. ⏎  ⏎ - On `main` (after #49193): ⏎   ``` ⏎   ValueError: There is no module or parameter named 'k_proj' in ⏎   MiniCPMV4_6ViTWindowAttentionSelfAttn. The available parameters ... are: ⏎   {'qkv_proj.weight', 'qkv_proj.bias', 'out_proj.weight', 'out_proj.bias'} ⏎   ``` ⏎ - On released `0.25.x` (after #47058): the same code path in …[truncated]

### L3-b3124a8237  (L3, 2026-09-14, sha b3124a8237f2, PR #56071)
TITLE: [Model] Add support for Nanbeige4.2 (transformers backend) (#56071)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/models/supported_models.md (+1/-0); tests/models/registry.py (+4/-0); vllm/model_executor/models/registry.py (+1/-0)
LABELS: documentation, ready, verified
BODY: ## Purpose ⏎ Add **transformers backend** support for Nanbeige 4.2 (NanbeigeForCausalLM): ⏎ ## Test Plan ⏎ ```bash ⏎ vllm serve Nanbeige/Nanbeige4.2-3B \ ⏎   --trust-remote-code \ ⏎   --model-impl transformers \ ⏎   --host 0.0.0.0 \ ⏎   --port 8001 \ ⏎   --served-model-name nanbeige4.2-3b \ ⏎   --reasoning-parser qwen3 \ ⏎   --enable-auto-tool-choice \ ⏎   --tool-call-parser qwen3_coder \ ⏎   --tensor-parallel-size 2 ⏎ ``` ⏎ ## Test Result ⏎ <img width="1483" he …[truncated]

### L3-3bb0a03f35  (L3, 2026-09-14, sha 3bb0a03f3526, PR #52228)
TITLE: [Model Runner V2] Acceptance estimation for adaptive verification (#52228)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/spec_decode/test_acceptance_estimator.py (+294/-0); tests/v1/worker/test_gpu_autoregressive_speculator.py (+5/-0); tests/v1/worker/test_gpu_extract_hidden_states_speculator.py (+2/-0); vllm/config/speculative.py (+10/-2); vllm/v1/worker/gpu/model_runner.py (+4/-0); vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py (+501/-0); vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py (+6/-0); vllm/v1/worker/gpu/spec_decode/dspark/speculator.py (+16/-18); vllm/v1/worker/gpu/spec_decode/speculator.py (+62/-8)
LABELS: speculative-decoding, ready, mrv2, dflash
BODY: # Context ⏎ Currently, adaptive verification is only enabled for DSpark speculators that have a confidence head. This PR enables it for ALL draft-model-based speculator types (MTP, EAGLE3, DFlash 1&2), including DSpark when a confidence head is not available. ⏎  ⏎ NOTE: The attention backend(s) of the base model must still support variable length verification, which is the existing prerequisite for DSpark adaptive verification. This PR depends on #5 …[truncated]

### L3-ba2ae9f239  (L3, 2026-09-14, sha ba2ae9f23961, PR #56666)
TITLE: [MRV2] Run pooling post processing on non-final PP ranks (#56666)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu_worker.py (+2/-0)
LABELS: ready, mrv2
BODY: ## Purpose ⏎  ⏎ Fix incorrect pooling outputs when Model Runner V2 uses pipeline parallelism and a request spans multiple scheduler steps. `Worker.execute_model()` previously called `pool()` only when the model forward returned `None`; non-final PP ranks instead return `IntermediateTensors`, so their pooling post-step never advanced `num_computed_tokens`. Later chunks consequently used stale request state. This change runs that post-step after enqu …[truncated]

### L3-00972dfd72  (L3, 2026-09-14, sha 00972dfd7298, PR #56513)
TITLE: [ROCm][DSV4.1][Perf] Fold the mHC post step into the delayed pre projection (#56513)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/test_mhc_kernels.py (+76/-2); vllm/_aiter_ops.py (+138/-37); vllm/model_executor/kernels/mhc/aiter.py (+20/-1); vllm/model_executor/layers/mhc.py (+75/-7); vllm/models/deepseek_v41/amd/model.py (+37/-17)
LABELS: rocm, ready, deepseek, DSv4.1
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ DeepSeek-V4.1-Flash on AMD runs an mHC post block and then the next sublayer's pre block at every seam. AITER can apply the post step and project the resulting residual in one kernel (`mhc_fused_post_pre_gemm_sqrsum`), which removes one launch per seam. ⏎  ⏎ `deepseek_v4/amd/model.py` already reaches that kernel through `MHCFusedPostPreOp`. V4.1 could not, for the same reason it could not use AITER's plain `mhc_pre`: it collapses each sub …[truncated]

### L3-dc89fdfb0e  (L3, 2026-09-14, sha dc89fdfb0ede, PR #56016)
TITLE: [CPU][Profiler] Group torch profiler tables by input shape when record_shapes is on (#56016)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/profiler/wrapper.py (+8/-2)
LABELS: verified
BODY: ## Purpose ⏎  ⏎ Group PyTorch profiler `key_averages` tables by input shape on the CPU platform when the user enables `--profiler-config.torch_profiler_record_shapes=true`. ⏎  ⏎ Without this, CPU dumps collapse every call of an op into one row, which hides prefill vs decode and GEMM size differences. GPU dumps are unchanged: `group_by_input_shape` is only set when `current_platform.is_cpu()` and `torch_profiler_record_shapes` are both true. Recording …[truncated]

### L3-79f0be21ff  (L3, 2026-09-14, sha 79f0be21ff40, PR #56773)
TITLE: [Bugfix][CPU] Fix DeepSeek-R1 (FP8 MLA + MoE) correctness on CPU backend (#56773)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/kernels/linear/scaled_mm/cpu.py (+2/-0); vllm/model_executor/layers/fused_moe/experts/cpu_moe.py (+1/-0)
LABELS: bug, deepseek, cpu
BODY: ## Purpose ⏎  ⏎ Fixes two independent bugs blocking `deepseek-ai/DeepSeek-R1` (and any other DeepSeekV3-routed / FP8 MLA model) on vLLM's CPU backend. ⏎  ⏎ 1. `CPUExpertsFp8._supports_routing_method` didn't allowlist `RoutingMethodType.DeepSeekV3`, so the model fails to load with `ValueError: Unsupported FP8 MoE backend: NONE`. ⏎ 2. `CPUFp8BlockScaledMMKernel.process_weights_after_loading` didn't honor `layer._cpu_skip_gemm_dispatch` (set by MLA's "absorb" …[truncated]

### L3-dc2e8f1157  (L3, 2026-09-14, sha dc2e8f115717, PR #54965)
TITLE: [ROCm][Perf] W4A16: keep skinny GEMM zero-points packed 4-bit (#54965)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/rocm/skinny_gemms_int4.cu (+41/-18); csrc/rocm/torch_bindings.cpp (+2/-1); tests/kernels/quantization/test_rdna_hybrid_w4a16.py (+55/-10); vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py (+19/-16)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ `RDNAHybridW4A16LinearKernel` expanded the asymmetric zero-points to the ⏎ activation dtype at load time (e.g. bf16 instead of int4), costing 4x the DRAM traffic. ⏎  ⏎ This PR keeps them in the packed form (8x int4 inside an int32). ⏎  ⏎ ### Performance ⏎  ⏎ | Model | Metric | before | after | delta | ⏎ | --- | --- | --- | --- | --- | ⏎ | cyankiwi/gemma-4-31B-it-AWQ-4bit | Median TPOT | 76.37 ms | 56.20 ms | **-26.4%** | ⏎ | cyankiwi/gemma-4- …[truncated]

### L3-e85c8826ce  (L3, 2026-09-14, sha e85c8826ce2a, PR #55650)
TITLE: [Bugfix] Make KV cache and MFU log lines backend-neutral (#55650)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/core/kv_cache_utils.py (+3/-1); vllm/v1/metrics/loggers.py (+6/-2); vllm/v1/metrics/perf.py (+1/-1)
LABELS: bug, ready, kv-cache-manager
BODY: ## Purpose ⏎  ⏎ Three log lines hardcode `"GPU"` regardless of the active backend, which is misleading on ⏎ the CPU backend (and on XPU/NPU). This addresses items 1–3 of #55614. ⏎  ⏎ The label is now derived from `device_config.device_type`, which is already resolved at ⏎ config construction and available at every call site — no new imports. ⏎  ⏎ **CUDA and ROCm output is unchanged**: `device_type == "cuda"` yields the same `"GPU"` ⏎ literal as before, an …[truncated]

### L3-5372e72a98  (L3, 2026-09-14, sha 5372e72a9884, PR #56633)
TITLE: [Perf][DSv4.1] Fold the mHC post block into the delayed pre projection (#56633)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/distributed/test_engram_dp_shard.py (+5/-2); tests/kernels/test_mhc_jit_warmup.py (+38/-9); tests/kernels/test_mhc_kernels.py (+244/-4); vllm/model_executor/kernels/mhc/tilelang.py (+283/-2); vllm/model_executor/kernels/mhc/tilelang_kernels.py (+124/-16); vllm/model_executor/kernels/mhc/warmup.py (+16/-6); vllm/models/deepseek_v41/nvidia/dspark.py (+1/-1); vllm/models/deepseek_v41/nvidia/model.py (+126/-52)
LABELS: ready, deepseek, DSv4.1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ DeepSeek-V4.1 runs an mHC post block and then the next sublayer's **delayed** pre block at every seam. Each seam therefore pays for three things: a post kernel, a split-k projection that re-reads the residual streams the post kernel just wrote, and the pre epilogue. ⏎  ⏎ DSv4 already avoids the middle read: `mhc_fused_tilelang` computes the updated residual in registers and feeds the pre-norm GEMM from there. V4.1 could not reuse it, for  …[truncated]

### L3-0a5747d410  (L3, 2026-09-14, sha 0a5747d410da, PR #51084)
TITLE: [Profiler] Add Proton CUDA graph attribution for MRV2 (#51084)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/contributing/profiling.md (+32/-14); tests/v1/worker/test_gpu_profiler.py (+393/-5); vllm/config/profiler.py (+16/-0); vllm/config/vllm.py (+39/-12); vllm/profiler/wrapper.py (+138/-11); vllm/v1/worker/gpu/model_runner.py (+10/-0); vllm/v1/worker/gpu_worker.py (+26/-3); vllm/v1/worker/mm_encoder_model_runner.py (+3/-0)
LABELS: documentation, performance, ready, nvidia, mrv2, verified
BODY: ## Purpose ⏎  ⏎ Add Proton CUDA graph replay attribution for the default model runner. With `proton_graph_attribution=true`, replayed kernels retain their capture context, including Triton launch metadata when the hook is enabled. Proton profiling now requires attribution whenever decoder or encoder CUDA graphs are enabled; otherwise replay kernels would be silently omitted. ⏎  ⏎ In short, this PR extends #48789 with proton cudagraph attribution capa …[truncated]

### L3-995e8581f4  (L3, 2026-09-14, sha 995e8581f462, PR #51167)
TITLE: [Model] Voxtral Realtime: add support for `CUDAGraphMode.FULL_DECODE_ONLY` (#51167)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/multimodal/generation/test_voxtral_realtime.py (+67/-39); vllm/model_executor/models/voxtral_realtime.py (+9/-3); vllm/model_executor/models/whisper_causal.py (+98/-20)
LABELS: ready, needs-rebase, multi-modality, nvidia, mistral
BODY: This PR allows running Voxtral Realtime full CG graph for decode-only batches. ⏎ Currently decode steps are stuck on piecewise capture and pay the per-step launch overhead.  ⏎ The blocker is the whisper encoder's block-pooling attention builder: its `build` did a `copy.deepcopy` of the common ⏎ attention metadata and allocated a fresh `slot_mapping` every step, so the tensor addresses baked in at capture time were dead by replay. ⏎  ⏎ This makes that  …[truncated]

### L3-e6eb0d120c  (L3, 2026-09-14, sha e6eb0d120c48, PR #56893)
TITLE: [Model][DSv4.1] Store the whole KV in MXFP8 (FlashMLA V4.1 record) (#56893)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.flashmla_build, L3.mla.flashmla_sparse
FILES: cmake/external_projects/flashmla.cmake (+59/-26); vllm/v1/attention/backends/mla/flashmla_sparse.py (+11/-0); vllm/v1/attention/backends/mla/sparse_swa.py (+12/-6); csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu (+64/-30); csrc/libtorch_stable/ops.h (+1/-1); csrc/libtorch_stable/torch_bindings.cpp (+1/-1); tests/kernels/test_compressor_kv_cache.py (+130/-0); tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py (+98/-0); vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py (+39/-23); vllm/models/deepseek_v41/amd/dspark.py (+2/-0); (+4 more)
LABELS: performance, ci/build, deepseek, DSv4, DSv4.1
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎  ⏎ DeepSeek-V4.1 has been reusing DeepSeek-V4's paged `fp8_ds_mla` record. FlashMLA has a V4.1-specific one that quantizes the RoPE dims too; this switches DeepSeek-V4.1 onto it on SM100. ⏎  ⏎ | | V4 (`ModelType::V4`) | **V4.1 (`ModelType::V41`)** | ⏎ |---|---|---| ⏎ | data row | 448 B fp8 e4m3 NoPE + 128 B bf16 RoPE | **512 B fp8 e4m3, RoPE included** | ⏎ | scale row | 7 UE8M0 of 64 dims + 1 pad byte | **16 UE8M0, one per 32 dims (MXFP8)** | ⏎ | by …[truncated]

### L3-dabc4362b4  (L3, 2026-09-14, sha dabc4362b47a, PR #56628)
TITLE: [ROCm][DSV4.1][Perf] Stride the DSA decode candidate mask over the live context (#56628)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+118/-2); tests/kernels/attention/test_rocm_aiter_candidate_mask.py (+226/-0)
LABELS: rocm, ready, DSv4.1
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ `rocm_fp8_paged_mqa_logits` sizes its logits workspace as `(batch_size * next_n, max_model_len)` and hands the whole tensor to `apply_candidate_mask`, so the mask grid was sized by `max_model_len` rather than by the context actually resident. On DeepSeek-V4.1-Flash with `--max-model-len 1048576` that is 1024 programs per row when only the first ~128 have anything to do. ⏎  ⏎ The cost tracked row count and ignored context, which is the sig …[truncated]

### L3-1d0d1081c4  (L3, 2026-09-14, sha 1d0d1081c409, PR #53721)
TITLE: [ROCm][Connector] SWA+HMA-support in MoRI-IO connector (Gemma4) (#53721)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_moriio_connector.py (+228/-4); tests/v1/kv_connector/unit/test_moriio_routing.py (+6/-1); vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py (+4/-3); vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py (+170/-38)
LABELS: rocm, ready, kv-connector
BODY: ## Purpose ⏎  ⏎ Run PD with sliding window with hybrid memory allocator enabled (Gemma4). For simplicity: ⏎ - READ mode only.  ⏎ - Other hybrids (mamba etc.) need to disable HMA, as before ⏎  ⏎ **Breaking change:**  ⏎ - Running a hybrid model with MoRI WRITE mode will now require explicitly turning off HMA, and otherwise raise `NotImplementedError: MoRIIO WRITE mode does not support hybrid KV cache groups (sliding-window attention). Use READ mode, or pa …[truncated]

### L3-e0c04c7b4d  (L3, 2026-09-14, sha e0c04c7b4ddd, PR #53696)
TITLE: [Bugfix][Models] Fix OpenPangu sleep mode with static sinks (#53696)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention/static_sink_attention.py (+0/-2); vllm/model_executor/models/openpangu.py (+11/-7)
LABELS: bug, ready, mistral, kv-cache-manager
BODY: ## Purpose ⏎  ⏎ Follow-up to #53507 and part of #53343. This change isolates OpenPangu and the static-sink issues exposed by its level-2 sleep-mode coverage from the generic static-model-buffer PR. ⏎  ⏎ ## Changes ⏎  ⏎ - Register OpenPangu `param_sink_value` as a non-persistent model buffer so the existing level-2 buffer save/restore path preserves it. ⏎ - Remove the duplicate `CustomOp.__init__` call from `StaticSinkAttention`. `Attention.__init__` alr …[truncated]

### L3-f84b0c4bce  (L3, 2026-09-14, sha f84b0c4bceee, PR #56908)
TITLE: [Model Runner V2] Tolerate lack of pinned memory support (#56908)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/cpu/buffer_utils.py (+3/-5); vllm/v1/worker/gpu/buffer_utils.py (+37/-6)
LABELS: ready, cpu, mrv2
BODY: In MRV2 we use UVA with static pointers which requires pinned memory. ⏎  ⏎ This change allows it to work on platforms pinned memory isn't supported/enabled, in particular WSL.
