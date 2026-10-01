### L1-63833f8034  (L1, 2026-08-09, sha 63833f8034fb, PR #34081)
TITLE: config: business code no longer reads the published ServerArgs (#34081)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.flashinfer_cutedsl
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py (+5/-7); python/sglang/srt/hardware_backend/mlx/model_runner.py (+4/-2); python/sglang/srt/layers/attention/dsa/dsa_indexer.py (+2/-2); python/sglang/srt/layers/attention/dsa/utils.py (+5/-2); python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+13/-9); python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py (+6/-2); python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py (+4/-2); python/sglang/srt/layers/attention/linear/kernels/kda_ptx.py (+2/-4); python/sglang/srt/layers/attention/linear/lightning_backend.py (+6/-5); python/sglang/srt/layers/cp/base.py (+8/-5); (+25 more)
LABELS: Multi-modal, deepseek, apple-silicon
BODY: ## What ⏎  ⏎ The last field reads outside the resolution pipeline are gone: the read ratchet's baselines are **zero** over the whole package minus the modules that own the slot. Two shapes remained, and both got a named home in `runtime_context`, which is the module that owns the storage. ⏎  ⏎ **Derived members.** `mamba_cache_chunk_size`, `max_speculative_num_draft_tokens`, `use_mla_backend()`, `get_attention_backends()`, `get_model_config()` and `cuted …[truncated]

### L1-449f0da78f  (L1, 2026-08-09, sha 449f0da78fda, PR #33808)
TITLE: [Intel GPU] DeepSeek V4 15/N: Add silu_and_mul_clamp support to triton fused_moe for XPU (#33808)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+5/-3)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L1-e226bb711c  (L1, 2026-08-09, sha e226bb711c25, PR #33962)
TITLE: enable TRT-LLM for MiniMax M3 by preserving SwiGLU params (#33962)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.framework, L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/flashinfer_trtllm_moe.py (+18/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-0); python/sglang/srt/layers/moe/moe_runner/base.py (+1/-0); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+9/-0); python/sglang/srt/layers/quantization/fp8.py (+30/-0); python/sglang/srt/layers/quantization/fp8_utils.py (+6/-0); python/sglang/srt/model_executor/runner/flashinfer_autotune.py (+1/-2); python/sglang/srt/models/minimax_m3.py (+1/-0)
LABELS: run-ci, jit-kernel
BODY: ## Motivation ⏎  ⏎ ## Modifications ⏎  ⏎ now `--moe-runner-backend flashinfer_trtllm_routed` can run for M3. ⏎  ⏎ ``` ⏎ SGLANG_OPT_USE_MINIMAX_DENSE_SPARSE_DECODE=1 \ ⏎ sglang serve \ ⏎   --trust-remote-code \ ⏎   --model-path MiniMaxAI/MiniMax-M3-MXFP8 \ ⏎   --reasoning-parser auto \ ⏎   --tool-call-parser auto \ ⏎   --tp 4 \ ⏎   --attention-backend trtllm_mha \ ⏎   --moe-runner-backend flashinfer_trtllm_routed \ ⏎   --chunked-prefill-size 8192 \ ⏎   --mem-fraction-static 0.75 \ …[truncated]

### L1-accc51c6db  (L1, 2026-08-10, sha accc51c6dbe5, PR #34167)
TITLE: [DSA] Fix top-k v2 dropping non-primary ranks' output on CUDA 13.1+ (root cause for #33835) (#34167)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/dsv4/topk.py (+8/-5); python/sglang/kernels/jit/csrc/deepseek_v4/topk_v1.cuh (+35/-23); python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh (+20/-12); python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh (+48/-25)
LABELS: run-ci, jit-kernel, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ DeepSeek-V4 DSA decode can return a corrupt top-k from the fused small-batch ⏎ cluster kernel, which downstream sparse attention then dereferences as garbage KV ⏎ indices (the illegal memory access reported in #33835). ⏎  ⏎ The trigger is the **CUDA toolkit version, not the GPU architecture**: identical ⏎ source and identical hardware (H200, sm_90a) are correct when built with nvcc ⏎ 12.9/13.0 and wrong when built with 13.1/13.2/13.3. ⏎  ⏎ ## Root …[truncated]

### L1-c80a38edcd  (L1, 2026-08-10, sha c80a38edcd2c, PR #34321)
TITLE: [Fix] Pin `cuda-tile` to 1.6.0rc5 to unblock Python 3.10 x86_64 installs (#34321)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+2/-0)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎ - Pin `cuda-tile==1.6.0rc5` so resolution stops picking the incomplete `1.6.0rc6` prerelease ⏎  ⏎ ## Background ⏎ - `cuda-tile` is transitive: `sglang` -> `flashinfer_python[cu13]==0.6.15.post1` -> `cuda-tile>=1.4.0` ⏎ - On PyPI it is a `wheel_stub` placeholder sdist that fetches the real wheel from `pypi.nvidia.com` at build time ⏎ - `1.6.0rc6` was published without `cp310` and `cp313` linux x86_64 wheels, so the stub raises `RuntimeError: Didn …[truncated]

### L1-7c7326ccb3  (L1, 2026-08-10, sha 7c7326ccb328, PR #31700)
TITLE: Fix DeepSeek-V4/DeepSeek-V4-Pro DP-attention gather semantics (#31700)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v4.py (+7/-2); python/sglang/srt/models/deepseek_v4_nextn.py (+7/-2)
LABELS: bug, deepseek, run-ci
ISSUES: #31699 [Bug]: DeepSeek-V4(-Pro) DP-attention (data-parallel-size>1) produces numerically-garbage output when moe_a2a_backend=none and attn_tp_size>1 — dp_gather_partial used where data is already replicated
BODY: ## Motivation ⏎  ⏎ Fixes #31699. ⏎  ⏎ DeepSeek-V4 DP-attention produces numerically invalid output when ⏎ `moe_a2a_backend=none`, `data_parallel_size>1`, and `attn_tp_size>1`. ⏎  ⏎ By the time the MoE gather runs, `self_attn` has already reduced its output ⏎ across the attention-TP group. The hidden states are therefore replicated ⏎ across attention-TP ranks. ⏎  ⏎ The existing `dp_gather_partial` calls treat those tensors as unreduced partial ⏎ contributions …[truncated]

### L1-df986c4d5e  (L1, 2026-08-10, sha df986c4d5e97, PR #34199)
TITLE: Consolidate CUDA VMM allocation helpers (#34199)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/dwdp/layout.py (+1/-1); python/sglang/srt/layers/moe/dwdp/page_pool.py (+28/-18); python/sglang/srt/layers/moe/dwdp/transport.py (+31/-36); python/sglang/srt/layers/moe/dwdp/vmm.py (+0/-258); python/sglang/srt/layers/moe/dwdp/weight_buffer.py (+32/-39); python/sglang/srt/cuda_vmm_utils.py (+380/-26); python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py (+1/-21); python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py (+5/-5); python/sglang/srt/mem_cache/kv_vmm_backing.py (+30/-101); python/sglang/srt/utils/cuda_vmm_transport_utils.py (+69/-104); (+3 more)
LABELS: run-ci, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ CUDA VMM consumers duplicate allocation, handle selection, mapping, and teardown logic. A shared lifecycle implementation keeps those paths consistent and makes rollback and cleanup ownership explicit. ⏎  ⏎ ## Modifications ⏎  ⏎ - Move the CUDA VMM helpers to `sglang.srt.cuda_vmm_utils` and add shared reservation, mapping, handle-selection, and tensor-view primitives. ⏎ - Migrate custom all-reduce, DWDP, KV backing, and multimodal feature tr …[truncated]

### L1-b498f46271  (L1, 2026-08-10, sha b498f46271a5, PR #34326)
TITLE: [CI] Add MegaMoE runner compatibility alias (#34326)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+16/-0); python/sglang/srt/environ.py (+1/-1)
LABELS: documentation, deepseek
BODY: ## Summary ⏎  ⏎ - accept `--moe-runner-backend megamoe` through the existing `MOE_RUNNER_BACKEND_CHOICES` ⏎ - normalize the target-model runner alias before model-specific resolution to `moe_runner_backend=auto` and `moe_a2a_backend=megamoe` ⏎ - raise `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK`'s default from 1024 to 8192 ⏎ - keep the DeepSeek V4 Pro GB300 high-throughput nightly on `--moe-a2a-backend megamoe` with its explicit 8320 override ⏎  ⏎ ## …[truncated]

### L1-dd20826e0a  (L1, 2026-08-10, sha dd20826e0a87, PR #34220)
TITLE: [AMD] Preserve the AITER expert mask across torch_memory_saver pause/resume (#34220)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+5/-0); python/sglang/srt/utils/weight_checker.py (+1/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ On ROCm with AITER, `MoriEPDispatcher` and `DeepEPDispatcher` create an `expert_mask_gpu` at construction time to mark which experts are local to the current EP rank. ⏎  ⏎ The mask is allocated while the model is inside the memory-saver `WEIGHTS` region, so its GPU pages are managed by `torch_memory_saver`. However, the dispatcher is not an `nn.Module`, and `expert_mask_gpu` is only a plain attribute, so it is not included in ` …[truncated]

### L1-8c5d5f75bf  (L1, 2026-08-11, sha 8c5d5f75bf8b, PR #33312)
TITLE: Fix DSV4 DSpark shared expert loading (#33312)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v4_dspark.py (+17/-2); test/registered/unit/model_loader/test_runai_model_streamer_loader.py (+1/-0); test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py (+104/-3)
LABELS: deepseek, run-ci, bypass-fastfail
BODY: ## Summary ⏎  ⏎ #33889 moved shared-expert fusion to a per-runner decision installed from each model's entry class. The DSV4 target exposes that gate, but the standalone DSpark entry class did not, so the draft used the wrong layout and skipped its bundled shared-expert tensors. ⏎  ⏎ This makes DSpark use the DSV4 gate and retain the resolved layout when it is built. Shared experts stay separate by default, while explicitly forced fusion remains supp …[truncated]

### L1-704808ed27  (L1, 2026-08-11, sha 704808ed27ce, PR #34148)
TITLE: [MiniMax-H3] SubBlock: training-free block-sparse attention for the DiT (#34148)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/router.py (+256/-0); python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/README.md (+169/-0); python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/__init__.py (+15/-0); python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/kernels.py (+300/-0); python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse_attn.py (+407/-0); python/sglang/multimodal_gen/runtime/platforms/cuda.py (+39/-0); python/sglang/multimodal_gen/runtime/platforms/interface.py (+2/-0); python/sglang/multimodal_gen/test/unit/test_subblock_sparse_attention.py (+357/-0)
LABELS: documentation, run-ci, diffusion
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: # [MiniMax-H3] SubBlock sparse attention: training-free block sparsity for the DiT ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎ The MiniMax-H3 DiT spends most of its time in full attention over a 37.7k-token ⏎ video latent, and that share grows with resolution and duration. This PR adds ⏎ `--attention-backend subblock_sparse_attn`: a cheap estimator runs before attention and ⏎ picks, per (head, query block), which 64-token key blocks are worth attending, ⏎ and FlashInfer …[truncated]

### L1-a58fa0388e  (L1, 2026-08-11, sha a58fa0388e30, PR #33669)
TITLE: [Fix] Correct W4AFP8 DeepEP scaling and mode-specific dtypes (#33669)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:kernel-correctness-cases, body_keyword
ARTIFACT_HINTS: L1.ep.deepep_dispatcher, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_w4a8_moe.py (+3/-0); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+4/-0); python/sglang/srt/layers/moe/utils.py (+20/-8); python/sglang/srt/layers/quantization/w4afp8.py (+30/-6); test/registered/unit/layers/moe/test_w4afp8_deepep_dtype.py (+124/-0); test/registered/unit/layers/moe/test_w4afp8_deepep_post_reorder.py (+145/-0)
LABELS: run-ci, run-ci-extra
ISSUES: #33660 [Bug] W4AFP8 + DeepEP crashes at first inference: TypeError "missing 1 required positional argument: 'routed_scaling_factor'"
DEEP_STUDY: deep-study correctness case sglang:a58fa0388e: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=#23754
BODY: ## Motivation ⏎  ⏎ W4AFP8 with DeepEP had two independent failures in the same normal/prefill path. ⏎  ⏎ First, `cutlass_w4a8_moe_deepep_normal` called `deepep_post_reorder_triton_kernel` without the required `routed_scaling_factor`, causing the original first-request `TypeError` in #33660. The model forward applies routed scaling after the cross-rank DeepEP combine, so this rank-local reduction must receive the neutral value `1.0`; using the model facto …[truncated]

### L1-d82a1d4802  (L1, 2026-08-11, sha d82a1d480241, PR #33905)
TITLE: [XPU] Pad MoE expert weight row stride to avoid L3 aliasing (#33905)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/utils.py (+35/-0); python/sglang/srt/layers/quantization/unquant.py (+70/-8); test/registered/xpu/test_moe_ld_padding.py (+180/-0)
LABELS: quant, intel, xpu, run-ci
BODY: apply kernel side changes in https://github.com/sgl-project/sgl-kernel-xpu/pull/187 ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎ Padding the MoE weight for specfic shapes on XPU ⏎  ⏎ ## Modifications ⏎  ⏎ Weight paddings during model loading ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get appr …[truncated]

### L1-6f3fe13a9c  (L1, 2026-08-11, sha 6f3fe13a9c81, PR #34364)
TITLE: [AMD] Install AITER's pinned Triton wheel in the ROCm 7.2 image (#34364)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/rocm.Dockerfile (+16/-23)
LABELS: amd
BODY: ## Motivation ⏎  ⏎ The ROCm 7.2 image builds Triton from source at its own pinned commit, leaving a 9.9GB checkout in the image. ⏎  ⏎ Install AITER pinned Triton during image build. ⏎  ⏎ ## Modifications ⏎  ⏎ `docker/rocm.Dockerfile` only, and only the two `-rocm720` stages, so ROCm 7.0 is byte-for-byte unchanged: call AITER's `.github/scripts/install_triton.sh` instead of building from source, so the image follows AITER's pin rather than carrying its ow …[truncated]

### L1-93e9db5eb8  (L1, 2026-08-11, sha 93e9db5eb89d, PR #34447)
TITLE: [Fix][Qwen]: fused shared-expert detection PP-safe protection (#34447)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/qwen3_5.py (+6/-6); test/registered/unit/models/test_qwen3_5_pipeline_parallel.py (+69/-0)
LABELS: high priority, run-ci, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎ The Qwen MoE fused shared-expert lookup always reads `self.model.layers[0].mlp`. `make_layers()` leaves decoder layers owned by other pipeline stages as `PPMissingLayer` placeholders, so a non-first PP stage fails during model initialization with: ⏎  ⏎ ```text ⏎ AttributeError: 'PPMissingLayer' object has no attribute 'mlp' ⏎ ``` ⏎  ⏎ ## Modifications ⏎  ⏎ - Scan `[self.model.start_layer, self.model.end_layer)` when detecting fused share …[truncated]

### L1-a3bd7d9401  (L1, 2026-08-12, sha a3bd7d94011d, PR #34372)
TITLE: Bump CuTeDSL to 4.6.2 (#34372)
SOURCES: dependency_pin, release_notes
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+2/-2); python/sglang/kernels/ops/attention/flash_attn/cute/interface.py (+0/-2); python/sglang/kernels/ops/attention/flash_attn/cute/pyproject.toml (+3/-3)
LABELS: high priority, dependencies, run-ci, jit-kernel, bypass-fastfail, run-ci-extra, release-highlight
BODY: ## Motivation ⏎  ⏎ This fixes an FA4 startup regression on Blackwell caused by the combination of `quack-kernels==0.6.3` and `nvidia-cutlass-dsl==4.6.0`. ⏎  ⏎ I hit this while starting `thinkingmachines/Inkling-Small-NVFP4` with `--attention-backend fa4` on SM103. During FlashInfer autotuning, the FA4 relative-bias kernel failed to compile: ⏎  ⏎ ```text ⏎ error[TYPE_UNSTABLE_JOIN]: `tBrS_cur` has type `None` on one path and `_Tensor` on another ⏎   --> f …[truncated]

### L1-00e57d74f0  (L1, 2026-08-12, sha 00e57d74f07b, PR #33997)
TITLE: Bump FlashInfer to 0.6.17 and remove Kimi K3 workarounds (#33997)
SOURCES: path_core, symbol_pickaxe, dependency_pin, release_notes
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+3/-49); docker/kimi_k3/flashinfer-perkz-dcp-0.6.15.txt (+0/-5639); docker/kimi_k3/kimi_k3_cu12.Dockerfile (+10/-47); docker/kimi_k3/kimi_k3_cu13.Dockerfile (+10/-47); python/pyproject.toml (+1/-1); python/sglang/kernels/ops/moe/trtllm_gen_moe.py (+0/-528); docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx (+2/-11); docs/src/snippets/_playground.jsx (+1/-1); docs/src/snippets/configs/moonshotai/kimi-k3.jsx (+3/-7); python/sglang/kernels/jit/csrc/kimi_k3/comm/ar_fusion.cuh (+1/-1); (+9 more)
LABELS: documentation, high priority, dependencies, run-ci, jit-kernel, release-highlight
BODY: ## Summary ⏎  ⏎ Bump FlashInfer to 0.6.17 and remove Kimi K3 workarounds ⏎  ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #31570458219](https://github.com/sgl-project/sglang/actions/runs/31570458219) ⏎ Latest PR Test (Extra): :x: [Run #31570457937](https://github.com/sgl-project/sglang/actions/runs/31570457937)

### L1-8e7c07fae7  (L1, 2026-08-12, sha 8e7c07fae734, PR #34587)
TITLE: [Docs] Add Qwen3.8 cookbook (#34587)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .claude/skills/cookbook-add-model/references/authoring-reference.md (+1/-1); .claude/skills/cookbook-add-model/templates/config.jsx.tmpl (+7/-3); docs/cookbook/autoregressive/Qwen/Qwen3.6.mdx (+0/-1); docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx (+315/-0); docs/cookbook/autoregressive/intro.mdx (+1/-1); docs/docs.json (+1/-0); docs/src/snippets/_playground.jsx (+18/-0); docs/src/snippets/configs/Qwen/qwen3.8-benchmarks.jsx (+23/-0); docs/src/snippets/configs/Qwen/qwen3.8.jsx (+939/-0)
LABELS: documentation
BODY: ## Motivation ⏎  ⏎ Qwen3.8-2.4T-A95B has no cookbook page. It is Qwen's largest open-weight model to date — 2.4T total / 95B active parameters, 92 layers as 23 repeats of `3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE)` — and its deployment story is unlike the dense-attention models already covered: two thirds of the layers are linear-attention, so the GDN recurrent-state pool rather than the KV cache is what caps concurrency, and at 2.4T p …[truncated]

### L1-197832bcf5  (L1, 2026-08-12, sha 197832bcf536, PR #33465)
TITLE: [Kimi-K3][NPU]  Support Kimi-K3 on NPU (#33465)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/hardware_backend/npu/moe/activation.py (+33/-0); python/sglang/srt/layers/moe/moe_runner/ascend.py (+16/-3); python/sglang/kernels/ops/elementwise/add3.py (+3/-1); python/sglang/kernels/ops/kimi_k3/__init__.py (+11/-6); python/sglang/kernels/ops/kimi_k3/mla_output_gate.py (+4/-1); python/sglang/kernels/ops/speculative/dspark/dspark_verify_window.py (+2/-1); python/sglang/multimodal_gen/runtime/layers/attention/backends/ascend_fa.py (+6/-0); python/sglang/srt/arg_groups/speculative_hook.py (+8/-4); python/sglang/srt/environ.py (+10/-0); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+46/-4); (+14 more)
LABELS: documentation, quant, dependencies, Multi-modal, deepseek, speculative-decoding, blackwell, npu, run-ci, diffusion
BODY: ## Summary ⏎  ⏎ Add Kimi-K3 support for Ascend NPU on top of upstream SGLang main and the GPU model integration from #32541. ⏎  ⏎ This PR keeps shared/GPU behavior intact where possible and dispatches NPU-specific implementations through the Ascend backend. The extracted Ascend Triton kernels are proposed separately in sgl-project/sgl-kernel-npu#658. ⏎  ⏎ ## Main changes ⏎  ⏎ - Full 93-layer Kimi-K3 execution on Ascend with TP64/DP4. ⏎ - DeepEP and Ascend …[truncated]

### L1-d44c836cfd  (L1, 2026-08-13, sha d44c836cfdcf, PR #34567)
TITLE: [XPU][CI] disable SYCL_CACHE_PERSISTENT to fix topk segfault (#34567)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: scripts/ci/xpu/xpu_ci_start_container.sh (+4/-1)
LABELS: intel, ci, xpu, run-ci, run-ci-extra
BODY: ## Summary ⏎  ⏎ Intel SYCL runtime's persistent kernel cache mishandles torch 2.13 XPU `aten.topk` on the pinned Intel graphics stack (compute-runtime 26.05 / IGC 2.28). Reloading the cached kernel segfaults inside `libsycl`, crashing `test_biased_grouped_topk` with SIGSEGV. ⏎  ⏎ Setting `SYCL_CACHE_PERSISTENT=0` in the CI container avoids the broken reload path. ⏎  ⏎ ## Repro ⏎  ⏎ Fresh `intel/deep-learning-essentials:2026.0.0` container, only `torch==2.13.0+xp …[truncated]

### L1-bca8ed4afc  (L1, 2026-08-13, sha bca8ed4afc03, PR #32941)
TITLE: [minimax m3][npu]Adaptation of Minimax M3(w8a8) for NPU platforms [1/2] (#32941)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/hardware_backend/npu/moe/activation.py (+37/-7); python/sglang/srt/hardware_backend/npu/moe/topk.py (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+7/-0); python/sglang/srt/layers/moe/moe_runner/ascend.py (+5/-1); python/sglang/srt/environ.py (+14/-0); python/sglang/srt/hardware_backend/npu/memory_pool_npu.py (+131/-0); python/sglang/srt/hardware_backend/npu/modules/minimax_m3_processor.py (+301/-0); python/sglang/srt/layers/attention/minimax_sparse_backend.py (+1201/-141); python/sglang/srt/layers/layernorm.py (+11/-3); python/sglang/srt/layers/quantization/modelslim/modelslim.py (+12/-0); (+8 more)
LABELS: documentation, quant, amd, dependencies, lora, Multi-modal, deepseek, speculative-decoding, sgl-kernel, blackwell
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Adaptation of Minimax M3 for NPU platforms ⏎  ⏎ ## Modifications ⏎  ⏎ **1、Define the complete attention implementation for MiniMax-M3 on the NPU platform, with deep optimization based on NPU‑specific features. ⏎ 2、Implement speculative inference adaptation for MiniMax‑M3 based on the Eagle3 draft model. ⏎ 3、Fix the W8A8 weight loading method for MiniMax‑M3. ⏎ 4、Fix the memory allocation issue in CUDA graph where the number of predicted  …[truncated]

### L1-ef7208d41d  (L1, 2026-08-13, sha ef7208d41d8e, PR #33559)
TITLE: [kernel] add triton moe TMA up support (#33559)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.triton
FILES: python/sglang/srt/layers/moe/moe_runner/triton.py (+3/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=512,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+164/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=512,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json (+164/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+29/-6); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py (+309/-80)
LABELS: run-ci, jit-kernel, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ Following PR #10567 ("Opt fused triton moe: add tma for down proj kernel"), which added TMA support for the down-projection (second MoE GEMM), this PR extends TMA support to the up-projection (first MoE GEMM, a.k.a. gate_up). ⏎  ⏎  ⏎ Runtime couples up/down TMA via c_sorted = down_moe_use_tma. Since c_sorted is a tl.constexpr, different values produce different kernel binaries. The old tuning script always used c_sorted=False for up …[truncated]

### L1-3f6ef01322  (L1, 2026-08-13, sha 3f6ef01322ca, PR #31956)
TITLE: Optimize MiniMax-M2.7 on CPU (#31956)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py, L1.hardware.cpu_npu_musa
FILES: python/sglang/kernels/aot/csrc/cpu/topk.cpp (+99/-50); python/sglang/srt/layers/moe/topk.py (+18/-11); python/sglang/kernels/aot/csrc/cpu/norm.cpp (+257/-0); python/sglang/kernels/aot/csrc/cpu/torch_extension_cpu.cpp (+41/-6); python/sglang/srt/model_executor/cpu_graph_runner.py (+14/-1); python/sglang/srt/models/minimax_m2.py (+20/-4); test/registered/cpu/test_norm.py (+81/-0); test/registered/cpu/test_topk.py (+157/-0)
LABELS: sgl-kernel, intel, cpu, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Optimize MiniMax-M2.7 on CPU [#issue 26439](https://github.com/sgl-project/sglang/issues/26439) ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - Add `topk_softmax_cpu` and `topk_sigmoid_cpu` kernels with `correction_bias`, optional renormalization, and FP32 gating output support, and dispatch to them from `fused_topk_cpu`. ⏎ - Make CPU routing consistently return `int32` `topk_ids` and remove the redundant `int64`-to-`int32` conversion from `fuse …[truncated]

### L1-9d34c2809f  (L1, 2026-08-13, sha 9d34c2809f58, PR #28354)
TITLE: [FlashInfer v0.6.16] Support FlashInfer CuTe DSL NVFP4 MoE quantization (#28354)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.runner.flashinfer_cutedsl, L1.runner.flashinfer_cutlass
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py (+125/-15); python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py (+3/-0); docs/docs/references/environment_variables.mdx (+6/-1); python/sglang/srt/arg_groups/overrides.py (+2/-1); python/sglang/srt/configs/model_config.py (+1/-1); python/sglang/srt/environ.py (+4/-1); python/sglang/srt/layers/quantization/base_config.py (+3/-0); python/sglang/srt/layers/quantization/kv_cache.py (+2/-2); python/sglang/srt/layers/quantization/modelopt_quant.py (+25/-13); python/sglang/srt/layers/quantization/nvfp4_online.py (+17/-12); (+4 more)
LABELS: documentation, quant, deepseek, blackwell, run-ci, bypass-fastfail
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ @humansand ⏎  ⏎ - Add FlashInfer CuTe DSL v2 MoE support to `--quantization nvfp4_online`. ⏎   - Convert eligible BF16, FP16, or FP8 expert weights to NVFP4 at load time. ⏎   - Compute and forward online per-token FP32 activation scales. ⏎   - Support no A2A and FlashInfer A2A; both use CuTe DSL v2. ⏎ - Keep the quantization contract established by merged upstream work: ⏎   - `nvfp4_online`: online NVFP4 weight conversion with online per-token F …[truncated]

### L1-240a12b302  (L1, 2026-08-13, sha 240a12b302d2, PR #28666)
TITLE: [AMD] Fuse shared_expert_gate GEMV into the MoE append kernel (HIP/aiter) (#28666)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.helper_kernels
FILES: python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py (+74/-22); python/sglang/srt/models/qwen2_moe.py (+46/-16); test/registered/amd/accuracy/mi35x/test_qwen35_mxfp4_eval_mi35x.py (+249/-0); test/registered/moe/test_fused_append_shared_experts.py (+301/-0)
LABELS: amd, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ On the AITER shared-expert-fusion path, computing the fused shared-expert routing weight launches a standalone GEMV — `self.shared_expert_gate(hidden_states)`, a `[M, hidden] × [hidden, 1]` matrix-vector op (the `Cijk_…MT1x2x512…` kernel, ~8.9 µs in decode) — whose only output feeds the subsequent `_fused_append_shared_experts_with_weights_kernel`. At decode batch sizes this GEMV is pure kernel-launch overhead (≈0 TFLOPs), so it  …[truncated]

### L1-463981922c  (L1, 2026-08-13, sha 463981922ce6, PR #34809)
TITLE: [Cookbook] Add DeepSeek-V4-Pro-0813 (Pro Official) serving recipes (#34809)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx (+20/-6); docs/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx (+52/-0); docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx (+439/-3)
LABELS: documentation, deepseek
BODY: ## Motivation ⏎  ⏎ `deepseek-ai/DeepSeek-V4-Pro-0813` is not in the cookbook yet. This adds it as a `pro-official` variant with the three serving strategies, verified end-to-end on 4×GB300, plus derived B300 (TP=8) cells marked **not verified**. ⏎  ⏎ The 0813 checkpoint bundles a **DSpark** draft head, so the low-latency recipe uses `--speculative-algorithm DSPARK`. This matters because EAGLE fails silently on this checkpoint: it loads, serves, and retur …[truncated]

### L1-0772e79ee7  (L1, 2026-08-14, sha 0772e79ee719, PR #34755)
TITLE: [CI][PD] Pin nccl rendezvous port per side to fix flaky disaggregation tests (#34755)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/test/server_fixtures/disaggregation_fixture.py (+12/-1); test/registered/disaggregation/test_disaggregation_dsv4.py (+4/-0)
BODY: ## Motivation ⏎  ⏎ The `test_disaggregation_dsv4` PD test (and other PD tests sharing the disaggregation fixture) intermittently fail at server startup with: ⏎  ⏎ ``` ⏎ torch.distributed.DistNetworkError: The server socket has failed to listen on any ⏎ local network address. port: 35061, useIpv6: false, code: -98, name: EADDRINUSE, ⏎ message: address already in use ⏎   ... ⏎   File ".../srt/distributed/parallel_state.py", line 2242, in init_distributed_environment …[truncated]

### L1-22dde1dd5b  (L1, 2026-08-14, sha 22dde1dd5b56, PR #31554)
TITLE: [Docs] Fill GLM-5.2 H200 FP8 speed cells (low-latency, balanced); fix MTP notation (#31554)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/cookbook/autoregressive/GLM/GLM-5.2.mdx (+1/-1); docs/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx (+35/-4)
LABELS: documentation
BODY: ## Motivation ⏎  ⏎ The three H200 + FP8 speed cells for GLM-5.2 have been "pending re-measurement" since the acceptance-length-pinned methodology landed. We measured the low-latency and balanced cells on 8xH200 following that methodology, and while auditing the recipes found an MTP notation mismatch in the docs. ⏎  ⏎ ## Modifications ⏎  ⏎ 1. Fill all three pending **H200 + FP8** speed cells in `glm-5.2-benchmarks.jsx`: low-latency (c=1/c=16), balanced (c=64/ …[truncated]

### L1-03c1d58112  (L1, 2026-08-14, sha 03c1d5811264, PR #34150)
TITLE: perf: add H200 Triton MoE configs for E256 N512 (#34150)
SOURCES: path_config_only
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=256,N=512,device_name=NVIDIA_H200.json (+24/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=256,N=512,device_name=NVIDIA_H200_down.json (+26/-0)
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Summary ⏎  ⏎ - add Triton 3.7.1 BF16 fused-MoE configurations for `E=256`, `N=512` on NVIDIA H200 ⏎ - tune the gate/up and down projections separately ⏎ - enable TMA for the down projection at the tuned `M=512` and `M=1024` tiers ⏎ - preserve the existing heuristic configuration at `M=256` ⏎  ⏎ The target shape is used by LLaDA2.1-mini. Without these files, SGLang logs a missing-config warning and falls back to the generic Triton heuristic. ⏎  ⏎ ## Tuning metho …[truncated]

### L1-6eb941a34c  (L1, 2026-08-14, sha 6eb941a34cb1, PR #34844)
TITLE: [Spec] Support MegaMoE for DSpark under dp attention (#34844)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/arg_groups/speculative_hook.py (+16/-3); test/registered/spec/dspark/test_dspark_draft_path_default.py (+30/-0)
LABELS: deepseek, speculative-decoding, run-ci
BODY: ## Motivation ⏎  ⏎ `_handle_dspark` rejected every non-`none` `--moe-a2a-backend` when `--enable-dp-attention` is on, so DSpark could not run alongside an EP all-to-all dispatch. ⏎  ⏎ The lockstep an EP dispatch needs was already in place. The DSV4-MoE draft path all-gathers `global_num_tokens` in `_fill_dp_moe_sync_metadata`, and a rank with no local generation requests still enters the draft forward via `run_idle_participation`, so every EP rank calls  …[truncated]

### L1-3adbbec2fd  (L1, 2026-08-14, sha 3adbbec2fd82, PR #34789)
TITLE: [MoE] Route every trtllm-gen MoE call site through one PDL guard (#34789)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+9/-4); python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py (+5/-0); python/sglang/srt/layers/quantization/mxfp4.py (+7/-0)
BODY: ## Motivation ⏎  ⏎ PDL on the trtllm-gen MoE can leave its grid-dependency wait unreleased when another stream overlaps the launch. The stalled rank never reaches its next collective, so the entire TP group hangs. `6a1d1f4422` (landed in #31681) identified this and capped PDL by token count via `SGLANG_TRTLLM_MOE_PDL_MAX_TOKENS` — but only at the two fp4 call sites in `moe_runner/flashinfer_trtllm.py`. ⏎  ⏎ The mxfp4 call sites never passed `enable_p …[truncated]

### L1-b95a746948  (L1, 2026-08-14, sha b95a74694842, PR #32944)
TITLE: [MoE] Fuse swiglu moe up gemm epilogue (#32944)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels, L1.runner.triton
FILES: python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py (+72/-1); python/sglang/srt/layers/moe/moe_runner/triton.py (+5/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+56/-12); python/sglang/srt/layers/quantization/unquant.py (+62/-0); python/sglang/srt/environ.py (+7/-0); test/registered/kernels/ops/moe/test_fused_swiglu_epilogue.py (+207/-0)
LABELS: quant, run-ci, jit-kernel, run-ci-extra
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ **What the MoE activation costs today.** The triton fused-MoE path runs three steps per layer: an up-GEMM that writes `intermediate_cache1` at the full gate+up width `N`, a standalone `silu_and_mul` that reads that buffer and writes `intermediate_cache2` at width `N/2`, and a down-GEMM. The middle step is pure data movement dressed as compute — it round-trips the entire intermediate tensor through HBM to apply an elementwise func …[truncated]

### L1-f2ab6e306b  (L1, 2026-08-15, sha f2ab6e306b3d, PR #)
TITLE: config: the alias form of the runner-side instance read
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: 
PR_RECORD: missing (use git/gh if needed)
BODY: 

### L1-c87a2ced12  (L1, 2026-08-15, sha c87a2ced121a, PR #)
TITLE: test(step-12): state the bag contract as what resolution produced, and the skill rule that goes with it
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: 
PR_RECORD: missing (use git/gh if needed)
BODY: 

### L1-d8399af70c  (L1, 2026-08-15, sha d8399af70cf6, PR #34810)
TITLE: fix(qwen3): support DeepEP-class backends and early EPLB state (#34810)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/qwen3_moe.py (+11/-3)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ This PR adds two model-level compatibility paths to Qwen3 MoE for Mooncake EP and EPLB. ⏎  ⏎ 1. Mooncake implements the DeepEP-class MoE dispatch interface, but Qwen3 only checks `get_moe_a2a_backend().is_deepep()`. As a result, Qwen3 skips the DeepEP-class setup and forward path for Mooncake. DeepSeek's shared-expert fusion work consolidated this backend policy behind `is_deepep_class_backend()` in #20089, providing the model-leve …[truncated]

### L1-7769f54feb  (L1, 2026-08-15, sha 7769f54febc9, PR #34883)
TITLE: [Kimi-K3] Use explicit SiTU activation for MegaMoE (#34883)
SOURCES: symbol_pickaxe, dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/quantization/mxfp4.py (+9/-5); python/sglang/srt/models/kimi_k3.py (+4/-16); test/registered/models_e2e/test_kimi_k3_b300.py (+54/-2)
LABELS: dependencies
BODY: ## Summary ⏎  ⏎ - remove the Kimi-K3 MegaMoE activation-clamp sentinel ⏎ - call DeepGEMM with `activation="situ"` directly ⏎ - prepare MXFP4 weights for MegaMoE even when Kimi-K3 selects `flashinfer_mxfp4` as its regular MoE runner ⏎ - replace the mock MegaMoE unit test with an 8-GPU B300 serving test that covers the real weight-loading, warmup, and GSM8K path ⏎  ⏎ ## Dependency ⏎  ⏎ Requires `sgl-deep-gemm==0.1.5.post3`, which includes https://github.com/sgl-proj …[truncated]

### L1-8d44091326  (L1, 2026-08-15, sha 8d4409132661, PR #34914)
TITLE: Update sgl-deep-ep release workflow for DeepEP v2 (#34914)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/release-whl-deepep.yml (+3/-3); docker/sgl-deep-ep.Dockerfile (+17/-10)
BODY: ## Motivation ⏎  ⏎ DeepEP's release branches have changed. CUDA 13 now uses the rebased DeepEP v2 branch and requires NCCL Gin from NCCL 2.30.7. The old workflow branch mapping and CUDA 13 GDRCopy builder dependency are obsolete. ⏎  ⏎ ## Modifications ⏎  ⏎ - update the release matrix to use: ⏎   - CUDA 13 x86_64/aarch64: `sgl-deepep` ⏎   - CUDA 12.9 x86_64: `sgl-deepep-cu12-x86` ⏎   - CUDA 12.9 aarch64: `sgl-deepep-cu12-arm` ⏎ - force reinstall `nvidia-nccl-cu13==2. …[truncated]

### L1-e161bd1265  (L1, 2026-08-15, sha e161bd1265a0, PR #34937)
TITLE: Fix Python packaging shadowing in DeepEP wheel builds (#34937)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: scripts/build_sgl_deepep.sh (+3/-3)
BODY: ## Summary ⏎  ⏎ - mount the DeepEP packaging overlay at `/sgl-deep-ep-packaging` instead of `/packaging` ⏎ - update the in-container build command to use the non-conflicting path ⏎  ⏎ ## Root cause ⏎  ⏎ The packaging overlay contains an `__init__.py` and was mounted at `/packaging`. Because the wheel container runs from `/`, that directory shadowed the installed Python `packaging` distribution. The CUDA 13 build therefore completed compilation and `auditwheel  …[truncated]

### L1-6314e9e4f5  (L1, 2026-08-15, sha 6314e9e4f5bd, PR #31794)
TITLE: [AMD][Fix] Qwen3.5: guard zero-grid launch in fused_qk_gemma_rmsnorm(_with_gate) (HIP invalid configuration on idle DP rank) (#31794)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/triton_gdn_fused_proj.py (+6/-0); python/sglang/srt/layers/attention/linear/gdn_backend.py (+8/-0); python/sglang/srt/layers/sampler.py (+4/-0); python/sglang/srt/models/qwen2_moe.py (+3/-2); python/sglang/srt/models/utils.py (+6/-0)
LABELS: run-ci, jit-kernel
ISSUES: #31350 [Bug] [AMD] Qwen3.5 + dp-attention: fused_qk_gemma_rmsnorm_with_gate Triton kernel aborts with HIP "invalid configuration argument" (zero-sized grid on empty/idle-rank batch) | #31594 [Bug] [AMD] Qwen3.5 GatedDeltaNet + dp-attention on ROCm/MoRI: HIP "invalid configuration argument" (linear-attn/Mamba state path); process hangs in chunk_gated_delta_rule_fwd under kernel serialization
DEEP_STUDY: deep-study correctness case sglang:6314e9e4f5: class=shape_alignment_edge; symptom=crash_or_exception; introducing=unknown
BODY: ## Motivation ⏎  ⏎ Bringing up **Qwen/Qwen3.5-397B-A17B-FP8** with MoRI expert-parallel all-to-all (`--moe-a2a-backend mori`) and `--enable-dp-attention` on MI355X/ROCm surfaces three connected failures. This PR fixes the two that live in the model / idle-rank path (bug #2 and bug #3 below); bug #1 is the prerequisite init fix already handled in #31793. ⏎  ⏎ **Bug #1 — Qwen shared-expert fusion vs. per-rank EP layout (prerequisite, fixed in #31793).* …[truncated]

### L1-66de161976  (L1, 2026-08-15, sha 66de1619760a, PR #32746)
TITLE: [Fix][AMD] MoRI EP: drop record_stream in TBO dispatch/combine (HSA out-of-resources) (#32746)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+10/-16)
LABELS: amd, run-ci
BODY: ## Motivation ⏎  ⏎ MoRI EP prefill with `--enable-two-batch-overlap` on MI355X aborts with `HSA_STATUS_ERROR_OUT_OF_RESOURCES`. The traceback blames whichever kernel needed scratch at that moment (for us, aiter `per_1x32_mx_quant_hip`), which is a victim, not the cause. ⏎  ⏎ The MoRI EP TBO path calls `record_stream(comm_stream)` on every dispatch/combine tensor. That parks each block in the caching allocator's deferred-free list until a comm-stream even …[truncated]

### L1-6ab4b99bc2  (L1, 2026-08-16, sha 6ab4b99bc255, PR #34962)
TITLE: [Quantization] Fix GPTQ scheme attachment broken by LinearBase.scheme default (#34962)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+5/-2); python/sglang/srt/layers/linear.py (+7/-2); python/sglang/srt/layers/quantization/gptq/gptq.py (+4/-4); python/sglang/srt/layers/vocab_parallel_embedding.py (+4/-0); test/registered/unit/layers/quantization/test_gptq_scheme_attach.py (+60/-0)
LABELS: run-ci, bypass-fastfail, run-ci-extra
BODY: ## Summary ⏎  ⏎ #29328 added a class-level `scheme = None` to `LinearBase`, turning GPTQ's `if not hasattr(layer, "scheme")` guard into a no-op — the scheme never gets built and every GPTQ checkpoint dies on load with `'NoneType' object has no attribute 'create_weights'`. The probe becomes `layer.scheme is None`, which makes that default load-bearing, so it's declared on the other two bases a GPTQ method can be handed (`VocabParallelEmbedding` for a  …[truncated]

### L1-0da87024d3  (L1, 2026-08-16, sha 0da87024d3a0, PR #30318)
TITLE: [NPU] Add mxfp4-w4a8 MOE Quantization Support for NPU (#30318)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/modelslim/modelslim.py (+4/-2); python/sglang/srt/layers/quantization/modelslim/schemes/__init__.py (+2/-0); python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_w4a8_mxfp4_moe.py (+83/-0); docs/docs/advanced_features/quantization.mdx (+2/-1); docs/docs/hardware-platforms/ascend-npus/optimization/quantization.mdx (+25/-0); python/sglang/srt/hardware_backend/npu/quantization/moe_methods.py (+86/-0)
LABELS: documentation, quant, npu, run-ci
BODY: ## Motivation ⏎  ⏎ SGLang already supports W4A8 MXFP quantization for linear layers on the Ascend NPU, but the corresponding MoE path is not yet supported. ⏎ This PR adds W4A8 MXFP support for ModelSlim-quantized MoE models. It also adapts the implementation to the latest refactored NPU MoE architecture, avoiding duplicated linear-layer support that is already available in the main branch. ⏎  ⏎ ## Modifications ⏎  ⏎ - Add ModelSlimMXFP4W4A8MoE to load M …[truncated]

### L1-56a759cffc  (L1, 2026-08-16, sha 56a759cffc2e, PR #34509)
TITLE: [JIT Kernel] Migrate moe_topk_softmax from AOT to JIT (#34509)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L1.routing.topk_softmax
FILES: python/sglang/kernels/jit/csrc/moe/moe_topk_softmax.cuh (+685/-0); python/sglang/kernels/ops/moe/__init__.py (+17/-2); python/sglang/kernels/ops/moe/moe_topk_softmax.py (+108/-0); test/registered/kernels/benchmark/moe/bench_moe_topk_softmax.py (+65/-0); test/registered/kernels/ops/moe/test_moe_topk_softmax.py (+200/-0)
LABELS: sgl-kernel, run-ci, jit-kernel, run-ci-extra
BODY: # [JIT Kernel] Migrate moe_topk_softmax from AOT to JIT ⏎  ⏎ Test and benchmark on an **H100 80GB (sm_90), CUDA 13.0, torch 2.12.0+cu130**. ⏎ **Size comparision** of the compiled files(.so) : ⏎ Kernel | Variants | JIT| AOT 1-arch | AOT ÷ JIT ⏎ -- | -- | -- | -- | -- ⏎ moe_topk_softmax | 3 dtypes | 379.69 KiB | 777.65 KiB | 2.048× ⏎   - JIT: one instantiated runtime variant. ⏎   - AOT: all variants combined into one .so for single arch. ⏎  ⏎  ⏎ ## Motivation …[truncated]

### L1-8e0499bd50  (L1, 2026-08-16, sha 8e0499bd5012, PR #31323)
TITLE: [AMD] [GLM5] Fuse shared-expert append into aiter grouped-topk (skip per-layer append kernel) (#31323)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+140/-18)
LABELS: amd, run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 35105 (confirmed_revert, reason=ci_or_test_failure) || deep-study performance PR (new_kernel_or_fusion)
BODY: ## Summary ⏎  ⏎ - On GLM-5.2 (`GlmMoeDsaForCausalLM`) with the non-EP aiter grouped-topk MoE route, every decoder layer runs a separate `_fused_append_shared_experts` kernel to append the shared expert into the top-k ids/weights. Instead, pre-populate a **persistent** top-k buffer's shared-expert columns once (`id = num_experts + i`, `weight = fused_shared_experts_scaling_factor`) and let the aiter kernel write only the routed columns via row strid …[truncated]

### L1-3adc70bb5e  (L1, 2026-08-16, sha 3adc70bb5e59, PR #34795)
TITLE: [MoE] Add H20 fp8_w8a8 tuned configs for Qwen3.8 (triton 3.7.1) + fix Qwen3_5MoeForCausalLM tuning (#34795)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=512,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); benchmark/kernels/fused_moe_triton/common_utils.py (+1/-0)
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Summary ⏎  ⏎ Two related changes for serving/tuning Qwen3.8 (`Qwen3_5MoeForCausalLM`) on NVIDIA H20: ⏎  ⏎ 1. **Tuned MoE kernel configs (main change)** ⏎    `E=512,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` in a new `configs/triton_3_7_1/` directory — the first configs for triton 3.7.1 (upstream pins `triton==3.7.1` via torch 2.13.0 on Linux; today the loader falls back to older-version dirs for every shape). Generated with …[truncated]

### L1-eb61cb2823  (L1, 2026-08-16, sha eb61cb28233b, PR #33480)
TITLE: [AMD] Support prefill context parallel two batch overlap for DeepSeek V4 (#33480)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/batch_overlap/operations_strategy.py (+24/-4); python/sglang/srt/batch_overlap/two_batch_overlap.py (+7/-0); python/sglang/srt/distributed/bootstrap.py (+6/-0); python/sglang/srt/distributed/parallel_state.py (+66/-0); python/sglang/srt/layers/attention/dsv4/compressor.py (+36/-0); python/sglang/srt/layers/dp_attention.py (+9/-0); python/sglang/srt/layers/utils/cp_utils.py (+50/-0); python/sglang/srt/models/deepseek_v4.py (+235/-15); python/sglang/srt/server_args.py (+6/-0); test/registered/amd/test_deepseek_v4_pro_fp4_cp_tbo.py (+155/-0)
LABELS: deepseek, run-ci, bypass-fastfail
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎ DeepSeek V4 prefill context parallelism (CP) distributes long-context prefill across attention CP ranks, but introduces several per-layer CP collectives around attention and MoE. Previously, the DeepSeek V4 two-batch-overlap (TBO) path only supported non-CP DP/EP workflows: ⏎ - `--enable-prefill-cp` with `--enable-two-batch-overlap` was rejected when DP attention was disabled. ⏎ - CP batches were explicitly excluded by the runtime TB …[truncated]

### L1-0099107e8b  (L1, 2026-08-16, sha 0099107e8b4a, PR #35105)
TITLE: Revert "[AMD] [GLM5] Fuse shared-expert append into aiter grouped-topk (skip per-layer append kernel)" (#35105)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+18/-140)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 31323 reason=ci_or_test_failure
BODY: Reverts sgl-project/sglang#31323 ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #32003209861](https://github.com/sgl-project/sglang/actions/runs/32003209861) ⏎ Latest PR Test (Extra): :no_entry_sign: [Run #32003209680](https://github.com/sgl-project/sglang/actions/runs/32003209680)

### L1-af743371cc  (L1, 2026-08-17, sha af743371cca9, PR #35060)
TITLE: Clean up environ.py: remove dead env vars, unify deprecation handling, move examples to a unit test (#35060)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/minimax_m2_5.mdx (+2/-2); docs/docs/references/environment_variables.mdx (+0/-10); python/sglang/srt/arg_groups/overrides.py (+0/-14); python/sglang/srt/disaggregation/common/staging_buffer.py (+3/-2); python/sglang/srt/environ.py (+129/-180); python/sglang/srt/server_args.py (+0/-34); test/registered/npu/accuracy/minimax_m2_5/test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms_gpqa.py (+2/-1); test/registered/npu/basic_function/speculative_inference/test_npu_speculative_moe_a2a_backend.py (+2/-1); test/registered/npu/performance/minimax_m2_5/test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py (+2/-1); test/registered/npu/performance/qwen3_235b_a22b/test_npu_qwen3_235b_w8a8_8p_in3k5_out1k5_50ms.py (+2/-1); (+2 more)
LABELS: documentation, speculative-decoding, npu, run-ci
BODY: ## Motivation ⏎  ⏎ `python/sglang/srt/environ.py` accumulated several kinds of clutter: deprecated descriptors nothing reads, four different mechanisms for warning about deprecated env vars, and ~80 lines of example functions at the bottom that were the module's only test coverage. ⏎  ⏎ ## Modifications ⏎  ⏎ **Remove dead descriptors** (verified no readers outside `environ.py`): ⏎ - `SGLANG_CUTLASS_MOE` — only consumer was the deprecation shim `_cutlass_moe_en …[truncated]

### L1-b6d7602914  (L1, 2026-08-17, sha b6d7602914d1, PR #22498)
TITLE: [CPU] Add support for Gemma4 on Xeon (#22498)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/aot/csrc/cpu/aarch64/moe.cpp (+6/-1); python/sglang/kernels/aot/csrc/cpu/moe.cpp (+19/-6); python/sglang/kernels/aot/csrc/cpu/moe.h (+20/-0); python/sglang/kernels/aot/csrc/cpu/moe_fp8.cpp (+7/-1); python/sglang/kernels/aot/csrc/cpu/activation.cpp (+1/-1); python/sglang/kernels/aot/csrc/cpu/extend.cpp (+35/-12); python/sglang/kernels/aot/csrc/cpu/gemm.h (+20/-0); python/sglang/kernels/aot/csrc/cpu/rope.cpp (+42/-2); python/sglang/kernels/aot/csrc/cpu/torch_extension_cpu.cpp (+13/-5); python/sglang/kernels/aot/csrc/cpu/vec.h (+8/-0); (+17 more)
LABELS: quant, Multi-modal, deepseek, sgl-kernel, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ This PR aims to add support for Gemma4 on Xeon. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ Basic functionality updates include: ⏎  ⏎ 1. Ported GPT-OSS PR #16775, which inclues sliding window attention support for intel_amx_attn and moe related interface updates. https://github.com/sgl-project/sglang/pull/22498/changes/be6c15b0654d3f9fe72bcfe0063729cdea3aec4e ⏎ 2. Added Moe support with Gelu activation: https://github.com/sgl-project/sglang/pull/2 …[truncated]

### L1-198a7b2fc9  (L1, 2026-08-17, sha 198a7b2fc990, PR #35062)
TITLE: [Misc] Clean up python/sglang package structure (#35062)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.fused_gate, L1.runner.marlin, L1.ep.other_dispatchers
FILES: python/sglang/kernels/ops/moe/moe_fused_gate.py (+1/-1); python/sglang/kernels/ops/moe/moe_wna16_marlin.py (+1/-1); python/sglang/README.md (+24/-16); python/sglang/__init__.py (+25/-44); python/sglang/_platform_stubs.py (+219/-9); python/sglang/_triton_stub.py (+0/-228); python/sglang/eval/llama3_eval.py (+0/-315); python/sglang/eval/loogle_eval.py (+0/-164); python/sglang/kernels/aot/python/sgl_kernel/debug_utils.py (+1/-1); python/sglang/kernels/fused_op.py (+1/-1); (+44 more)
LABELS: documentation, quant, hicache, sgl-kernel, run-ci, diffusion, jit-kernel
BODY: ## Motivation ⏎  ⏎ Clean up the top-level python/sglang package organization and remove stale modules. ⏎  ⏎ ## Modifications ⏎  ⏎ - Remove the obsolete eval package. ⏎ - Move frontend global configuration under sglang.lang and update internal imports. ⏎ - Move kernel API logging under sglang.kernels and update all callers. ⏎ - Simplify package initialization and move plugin loading to the top-level imports. ⏎ - Rewrite python/sglang/README.md to document the current …[truncated]

### L1-bc312d185d  (L1, 2026-08-17, sha bc312d185dc1, PR #34926)
TITLE: Clean deprecated DeepSeek V4 Environs (#34926)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.runner.deep_gemm, L1.runner.deepgemm_megamoe
FILES: python/sglang/srt/layers/moe/mega_moe.py (+18/-27); python/sglang/srt/layers/moe/mega_moe_sm90.py (+10/-36); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+5/-34); python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+6/-14); python/sglang/srt/layers/moe/topk.py (+4/-12); docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx (+1/-3); docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx (+1/-1); docs/docs/references/environment_variables.mdx (+0/-5); python/sglang/srt/arg_groups/deepseek_v4_hook.py (+1/-4); python/sglang/srt/arg_groups/overrides.py (+0/-6); (+18 more)
LABELS: documentation, deepseek, run-ci, run-ci-extra
DEEP_STUDY: deep-study: introduced the defect fixed in case sglang:74df026877 (fix PR 35677)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L1-b83d507cd7  (L1, 2026-08-17, sha b83d507cd711, PR #33676)
TITLE: [NPU] Support DeepSeek-V4 DSpark and refactor DSV4 cache management (#33676)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/speculative/dspark/dspark_verify_window.py (+6/-2); python/sglang/srt/arg_groups/speculative_hook.py (+2/-2); python/sglang/srt/disaggregation/ascend/conn.py (+27/-7); python/sglang/srt/disaggregation/decode.py (+2/-23); python/sglang/srt/disaggregation/prefill.py (+2/-3); python/sglang/srt/disaggregation/utils.py (+17/-17); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+24/-17); python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py (+545/-614); python/sglang/srt/hardware_backend/npu/dsv4/c128_sidecar_component.py (+294/-0); python/sglang/srt/hardware_backend/npu/dsv4/dsv4_allocator.py (+134/-425); (+27 more)
LABELS: deepseek, speculative-decoding, npu, run-ci, jit-kernel
BODY: ## Motivation ⏎  ⏎ This PR adds NPU support for DeepSeek-V4 DSpark speculative decoding. It also consolidates the DSV4 NPU cache and memory-pool implementation and includes several small execution-path optimizations. ⏎  ⏎ ## Summary of changes ⏎  ⏎ ### 1. DeepSeek-V4 DSpark support on NPU ⏎  ⏎ - Add ModelSlim W4A8/W8A8 DSpark checkpoint loading with NPU-specific weight remapping while preserving the existing CUDA loading path. ⏎ - Support checkpoint-local embeddin …[truncated]

### L1-fcdaaf8a5d  (L1, 2026-08-17, sha fcdaaf8a5d2c, PR #32313)
TITLE: [Feature] Optimize TP LMHead with All-to-All (#32313)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/arg_groups/overrides.py (+41/-1); python/sglang/srt/distributed/bootstrap.py (+53/-0); python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py (+4/-2); python/sglang/srt/layers/logits_processor.py (+80/-3); python/sglang/srt/server_args.py (+23/-4); test/registered/mock_model/test_e2e_tp.py (+24/-1); test/registered/unit/test_model_overrides.py (+1/-0)
LABELS: run-ci, release-highlight
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ When the model is deployed with pure DP and dp-attention, **--enable-dp-lm-head will suffer from poor gemm efficiency when batchsize of each rank is small**. For example, in DeepseekV4 Pro, we use dp=8 and --enable-dp-lm-head. In such case, the lmhead gemm itself costs **320us** when bs=36 on each dp rank. ⏎ <img width="557" height="122" alt="image" src="https://github.com/user-attachments/assets/c63cd93d-b672-4834-92d4-b046decfd6 …[truncated]

### L1-24d625698d  (L1, 2026-08-18, sha 24d625698d44, PR #31370)
TITLE: [AMD] feat(moe): fold padded-topk_ids fill into fused shared-experts append+remap (#31370)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.helper_kernels, L1.routing.topk_py
FILES: python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py (+19/-0); python/sglang/srt/layers/moe/topk.py (+17/-1); test/registered/moe/test_fused_append_remap_per_rank_shared_slots.py (+47/-0)
LABELS: amd, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Summary ⏎  ⏎ On the aiter **per-rank shared-slot** path (MoRI / DeepEP-class EP, EPLB off), the padded-`topk_ids` ⏎ region was zero-filled by a **separate `_fill_padded_rows` Triton launch** immediately before the ⏎ fused shared-experts **append + per-rank-shared-slot remap** kernel ran. ⏎  ⏎ This PR **folds that padded fill into the append+remap kernel**: rows `>= num_token_non_padded` get ⏎ `pad_fill_id` in every routed slot, computed on the rows a …[truncated]

### L1-cfc6dfb364  (L1, 2026-08-18, sha cfc6dfb3642b, PR #34923)
TITLE: Apply latest DeepEP branch (#34923)
SOURCES: path_core, path_integration+keyword, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L1.ep.deepep_dispatcher, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+6/-0); python/pyproject.toml (+1/-1); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+15/-0); scripts/ci/cuda/ci_install_dependency.sh (+12/-0); test/registered/4-gpu-models/test_deepseek_v3_cutedsl_4gpu.py (+4/-0)
LABELS: dependencies, deepseek, run-ci, run-ci-extra, release-highlight
BODY: ## Summary ⏎  ⏎ - configure `NVSHMEM_QP_DEPTH` before CUDA DeepEP low-latency buffer initialization ⏎ - enforce `max(existing value, 1024, 2 * (num_max_dispatch_tokens_per_rank + 1))` ⏎ - preserve larger user-provided values and leave NPU and normal DeepEP paths unchanged ⏎ - force reinstall `nvidia-nccl-cu13==2.30.7` in CUDA 13 CI and the main CUDA 13 image; CUDA 12 remains unchanged ⏎ - bump the runtime dependency to `sgl-deep-ep==0.1.1` ⏎  ⏎ ## Motivation ⏎  ⏎ De …[truncated]

### L1-3a8f522f65  (L1, 2026-08-18, sha 3a8f522f6547, PR #30612)
TITLE: install sglang in virtual env instead of system path (#30612)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+19/-21)
BODY: ## Motivation ⏎  ⏎ Install `sglang` and related packages in python virtual env instead of system path to avoid conflicts in updating pip packages and debian python packages.  ⏎  ⏎ Detailed error is:  ⏎ ``` ⏎ #26 473.6     Found existing installation: cryptography 41.0.7 ⏎ #26 473.6 error: uninstall-no-record-file ⏎ #26 473.6  ⏎ #26 473.6 × Cannot uninstall cryptography 41.0.7 ⏎ #26 473.6 ╰─> The package's contents are unknown: no RECORD file was found for  …[truncated]

### L1-9485c083bb  (L1, 2026-08-18, sha 9485c083bbc8, PR #30319)
TITLE: [NPU] Add mxfp4-w4a4 MOE Quantization Support for NPU (#30319)
SOURCES: path_core
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/hardware_backend/npu/moe/quant.py (+6/-5); docs/docs/advanced_features/quantization.mdx (+2/-1); docs/docs/hardware-platforms/ascend-npus/optimization/quantization.mdx (+25/-0); python/sglang/srt/hardware_backend/npu/quantization/moe_methods.py (+80/-0); python/sglang/srt/layers/quantization/modelslim/modelslim.py (+2/-0); python/sglang/srt/layers/quantization/modelslim/schemes/__init__.py (+2/-0); python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_w4a4_mxfp4_moe.py (+82/-0)
LABELS: documentation, quant, npu, run-ci
BODY: ## Motivation ⏎  ⏎ SGLang already supports ModelSlim W4A4_MXFP4 quantization for dense linear layers on Ascend NPU, but the corresponding MoE path is still missing. As a result, MoE models whose expert weights are exported with the W4A4_MXFP4 scheme cannot be loaded and executed through the ModelSlim quantization backend. ⏎ The Ascend MoE implementation has also been refactored into the modular AscendRunner architecture. Therefore, MXFP4 MoE support …[truncated]

### L1-ae6945e112  (L1, 2026-08-18, sha ae6945e11233, PR #35114)
TITLE: [kernels] Reorganize ops/diffusion by operator domain behind a lazy facade (#35114)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: .claude/skills/llm-torch-profiler-analysis/references/fuse-overlap-catalog.md (+7/-7); python/sglang/kernels/fused_op.py (+1/-0); python/sglang/kernels/ops/diffusion/README.md (+153/-0); python/sglang/kernels/ops/diffusion/__init__.py (+452/-96); python/sglang/kernels/ops/diffusion/activation/__init__.py (+1/-0); python/sglang/kernels/ops/diffusion/activation/sana_conv_post_triton.py (+1/-1); python/sglang/kernels/ops/diffusion/activation/silu_mul_bitexact.py (+1/-1); python/sglang/kernels/ops/diffusion/attention/__init__.py (+1/-0); python/sglang/kernels/ops/diffusion/attention/sana_wm_gdn_chunkwise_triton.py (+1/-1); python/sglang/kernels/ops/diffusion/attention/sana_wm_gdn_triton.py (+1/-1); (+157 more)
LABELS: documentation, blackwell, npu, run-ci, diffusion, apple-silicon, jit-kernel, run-ci-extra
BODY: ## Motivation ⏎  ⏎ `kernels/ops/diffusion` is the only operator group in `kernels/ops` organized by **backend** (`triton/`, `cutedsl/`, `flydsl/`) instead of by operator. Every sibling group puts the backend in the *filename* (`causal_conv1d_triton.py`, `cutedsl_bf16_gemm.py`, `mxfp8_moe_amd_gfx95.py`) and reserves subdirectories for algorithm families (`attention/{dsa,dsv4,fla,flash_attn,linear}`). ⏎  ⏎ The cost is concrete. `norm + scale/shift` has six …[truncated]

### L1-ccbe380028  (L1, 2026-08-19, sha ccbe38002873, PR #35407)
TITLE: [CI] Trim the base-c 4-gpu-h100 stage from 5 shards to 4 (#35407)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/test/kits/pd_parity_kit.py (+70/-0); test/registered/disaggregation/test_disaggregation_kimi_linear.py (+13/-75); test/registered/disaggregation/test_disaggregation_unified_memory.py (+16/-101); test/registered/ep/test_deepep_small.py (+1/-265); test/registered/ep/test_deepep_small_extra.py (+167/-0); test/registered/hicache/test_hicache_storage_3fs_backend.py (+1/-1); test/registered/models_e2e/test_qwen3_next_models.py (+1/-42); test/registered/models_e2e/test_qwen3_next_models_extra.py (+76/-0); test/registered/pp/test_pp_gemma4.py (+153/-0); test/registered/pp/test_pp_single_node.py (+1/-135); (+2 more)
LABELS: hicache
BODY: `base-c-test-4-gpu-h100` runs its shards serially (`max_parallel = 5 // 3 = 1`), so cutting `est_time` is a linear wall-clock win. ⏎  ⏎ **5470s / 16 files -> 3572s / 13 files, 5 shards -> 3.** Nothing is dropped from CI. ⏎  ⏎ Most of it is placement: several tests sat on the 4-GPU runner without needing four GPUs. ⏎  ⏎ - `test_flashinfer_comm_fusion.py` -> `base-b`/`1-gpu-small`, dropping its `4-gpu-b200` and `4-gpu-gb300` registrations. Collectives are fake …[truncated]

### L1-5f12839591  (L1, 2026-08-19, sha 5f128395910d, PR #35077)
TITLE: [Fix] Support Kimi-K3 ModelOpt mixed NVFP4/FP8 checkpoint (#35077)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+13/-2); python/sglang/srt/layers/quantization/modelopt_quant.py (+30/-11); python/sglang/srt/models/kimi_k3.py (+82/-23); test/registered/unit/layers/quantization/test_modelopt_nvfp4_moe_scales.py (+20/-0); test/registered/unit/model_loader/test_modelopt_loader.py (+22/-0); test/registered/unit/models/test_kimi_k3_bfa_overlap.py (+38/-2)
LABELS: quant, blackwell, run-ci, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ The official [nvidia/Kimi-K3-NVFP4](https://huggingface.co/nvidia/Kimi-K3-NVFP4) checkpoint is a ModelOpt mixed-precision checkpoint: ⏎  ⏎ - routed MoE experts use NVFP4 with SiTU (`beta=4`, `linear_beta=25`); ⏎ - supported attention projections use weight-only `FP8_PB_WO` with 128x128 block scales. ⏎  ⏎ Current `main` cannot serve this checkpoint with the FlashInfer TRT-LLM MoE backend. It rejects gated `situ` during startup, and it does no …[truncated]

### L1-03cf2de2e3  (L1, 2026-08-19, sha 03cf2de2e360, PR #35545)
TITLE: [Qwen3.5][MTP] Preserve online NVFP4 draft quantization for mixed checkpoints (#35545)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/model_config.py (+3/-3)
BODY: ## Motivation ⏎  ⏎ When a mixed ModelOpt checkpoint is used with an explicitly configured `nvfp4_online` speculative draft model, checkpoint detection currently overwrites the explicit draft quantization choice with `modelopt_mixed`. This prevents the MTP draft MoE from using the intended load-time NVFP4 conversion and FlashInfer CuTeDSL execution path. ⏎  ⏎ ## Modifications ⏎  ⏎ - Preserve an explicitly selected `nvfp4_online` draft quantization mode for bo …[truncated]

### L1-4f8ecf6ae9  (L1, 2026-08-19, sha 4f8ecf6ae9a8, PR #29525)
TITLE: [Feature] Add DeepEPv2 (ElasticBuffer) MoE A2A backend (#29525)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:confirmed-reverts(reverted), corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.framework, L1.runner.deep_gemm, L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/kernels/ops/moe/ep_moe_kernels.py (+402/-0); python/sglang/srt/layers/moe/ep_moe/layer.py (+7/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+10/-0); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+218/-1); python/sglang/srt/layers/moe/moe_runner/runner.py (+16/-0); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+8/-0); python/sglang/srt/layers/moe/token_dispatcher/base.py (+19/-0); python/sglang/srt/layers/moe/token_dispatcher/deepep_v2.py (+548/-0); python/sglang/srt/layers/moe/utils.py (+43/-3); python/sglang/srt/models/deepseek_v2.py (+12/-5); (+8 more)
LABELS: deepseek, jit-kernel, bypass-fastfail
DEEP_STUDY: deep-study: this PR was reverted by PR 35568 (confirmed_revert, reason=ci_or_test_failure) || deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ DeepEP v2's `ElasticBuffer` is an NCCL-symmetric-memory all-to-all engine with a ⏎ fixed per-rank capacity. Fixed capacity means static communication shapes, which ⏎ makes the MoE decode path CUDA-graph capturable under any topology — including ⏎ multi-node, where legacy `low_latency` is not available as a graphable path. ⏎ This PR adds it as a standalone backend `deepep_v2` alongside `deepep`; only ⏎ expert-parallel dispatch/combine is repl …[truncated]

### L1-746418a1ec  (L1, 2026-08-19, sha 746418a1ec78, PR #35041)
TITLE: [DSA] Trim top-k v2 output modes and tighten its PDL waits (#35041)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/dsv4/topk.py (+11/-4); python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh (+109/-77); python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh (+2/-8); python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py (+2/-4); test/registered/kernels/ops/attention/test_topk_v2.py (+30/-60)
LABELS: run-ci, jit-kernel, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: > Generated by Claude. ⏎  ⏎ ## Motivation ⏎  ⏎ The DeepSeek-V4 (DSA indexer) top-k v2 kernel carried an optional `raw_indices` ⏎ output that no production caller ever requests: the c4 indexer routes to the v1 ⏎ kernel whenever it needs raw indices, so the v2 argument was only ever exercised ⏎ by a unit test. Everyone else still paid for it with a per-element ⏎ `raw_out != nullptr` branch inside the page-table transform loop. ⏎  ⏎ While removing it, two PDL (programm …[truncated]

### L1-1270204d2c  (L1, 2026-08-19, sha 1270204d2c69, PR #35568)
TITLE: Revert "[Feature] Add DeepEPv2 (ElasticBuffer) MoE A2A backend" (#35568)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:confirmed-reverts
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.framework, L1.runner.deep_gemm, L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/kernels/ops/moe/ep_moe_kernels.py (+0/-402); python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-7); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+0/-10); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+1/-218); python/sglang/srt/layers/moe/moe_runner/runner.py (+0/-16); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+0/-8); python/sglang/srt/layers/moe/token_dispatcher/base.py (+0/-19); python/sglang/srt/layers/moe/token_dispatcher/deepep_v2.py (+0/-548); python/sglang/srt/layers/moe/utils.py (+3/-43); python/sglang/srt/models/deepseek_v2.py (+5/-12); (+8 more)
LABELS: deepseek, jit-kernel
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 29525 reason=ci_or_test_failure
BODY: Reverts sgl-project/sglang#29525 ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #32302852032](https://github.com/sgl-project/sglang/actions/runs/32302852032) ⏎ Latest PR Test (Extra): :no_entry_sign: [Run #32302852011](https://github.com/sgl-project/sglang/actions/runs/32302852011)

### L1-d216737e47  (L1, 2026-08-19, sha d216737e4768, PR #35372)
TITLE: [Kernel] Support wider rows in mega_moe_pre_dispatch (#35372)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/jit/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh (+81/-43)
LABELS: run-ci, jit-kernel
BODY: ## Motivation ⏎  ⏎ `mega_moe_pre_dispatch` assigns one thread to each 8-element input chunk, so a single CUDA block can only cover hidden dimensions up to 8192. Wider shapes are rejected before launch. ⏎  ⏎ ## Modifications ⏎  ⏎ - Cap the block size at the CUDA 1024-thread limit and process wider rows in strided chunks. ⏎ - Recompute quantization-group indices for each chunk while preserving the UE8M0 scale layout. ⏎ - Keep a separate single-chunk kernel special …[truncated]

### L1-c7478228dd  (L1, 2026-08-19, sha c7478228dd29, PR #30984)
TITLE: [AMD] [Docker] Upgrade Python 3.12 + torch 2.11 + triton 3.7 in ROCm 7.2.4 (#30984)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/rocm.Dockerfile (+199/-48); .github/workflows/pr-test-amd-extra.yml (+15/-3); .github/workflows/pr-test-amd-rocm720.yml (+46/-20); .github/workflows/release-docker-amd-rocm720-nightly.yml (+18/-3); .github/workflows/release-docker-amd.yml (+17/-4); python/pyproject_other.toml (+17/-0); scripts/ci/amd/amd_ci_install_dependency.sh (+88/-38); scripts/ci/amd/amd_ci_start_container.sh (+2/-2); scripts/ci/amd/amd_ci_start_container_disagg.sh (+26/-14)
LABELS: amd, dependencies, jit-kernel
BODY: ## Motivation ⏎  ⏎ Add ROCm 7.2.4 Docker flavors on Python 3.12 with PyTorch 2.11 and Triton 3.7.  ⏎  ⏎ PyTorch 2.11 for ROCm 7.2 is available from the PyTorch Foundation index. Its dependency initially installs `triton-rocm==3.6.0`, but this PR replaces it at the end of the build with AITER’s pinned Triton 3.7. Installing Triton last prevents later dependency resolution from reverting the validated ROCm stack. ⏎  ⏎ | Component | ROCm 7.2.0 flavors | R …[truncated]

### L1-b6dcd393d6  (L1, 2026-08-19, sha b6dcd393d6db, PR #35593)
TITLE: [Fix] Support 128-aligned hidden sizes in the W4AFP8 DeepEP low-latency requant kernel (#35593)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/kernels/ops/moe/ep_moe_kernels.py (+23/-5); test/registered/kernels/ops/moe/test_fp8_per_token_to_per_tensor_quant.py (+82/-0)
LABELS: quant, run-ci, jit-kernel
BODY: ## Motivation ⏎  ⏎ `fp8_per_token_to_per_tensor_quant_triton()` requantizes DeepEP low-latency's per-token-group ⏎ fp8 payload into the per-tensor fp8 that the first CUTLASS W4A8 grouped GEMM consumes. It ⏎ required the hidden size to fill a whole number of 1024-element blocks: ⏎  ⏎ ```python ⏎ K_BLOCK_SIZE = 1024 ⏎ assert x.size(2) % K_BLOCK_SIZE == 0 ⏎ grid = (x.size(2) // K_BLOCK_SIZE, 32, x.size(0)) ⏎ ``` ⏎  ⏎ so it rejected any expert hidden size that is only 128-a …[truncated]

### L1-1ef7882a5b  (L1, 2026-08-19, sha 1ef7882a5b76, PR #35294)
TITLE: [NIXL] Query EP top-k index dtype (#35294)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/nixl.py (+8/-1)
LABELS: run-ci
BODY: NIXL EP can be built with different top-k index types. Query the dtype reported by the installed nixl_ep instead of hardcoding torch.int64. ⏎  ⏎ ai-dynamo/nixl#1751 changes the default NIXL EP top-k index type from 64-bit to 32-bit. Following nixl_ep.topk_idx_t keeps SGLang compatible with both the upstream default and custom NIXL EP builds. Older builds that do not expose this attribute continue to use torch.int64. ⏎  ⏎ cc @ShangmingCai  ⏎  ⏎ --- ⏎ ### CI …[truncated]

### L1-50dae2d99d  (L1, 2026-08-19, sha 50dae2d99d70, PR #32340)
TITLE: Amd/dsv4 shared experts fusion top6 (#32340)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.helper_kernels, L1.routing.topk_py, L1.routing.fused_gate
FILES: python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py (+49/-23); python/sglang/kernels/ops/moe/moe_fused_gate.py (+9/-2); python/sglang/srt/layers/moe/topk.py (+13/-3); test/registered/kernels/ops/moe/test_moe_fused_gate.py (+36/-0); test/registered/moe/test_fused_append_remap_per_rank_shared_slots.py (+69/-3); test/registered/moe/test_fused_append_shared_experts_top6.py (+115/-0)
LABELS: amd, deepseek, run-ci, jit-kernel
BODY: # [AMD] DeepSeek-V4: fix shared-experts fusion for top-6 ⏎  ⏎ ## Summary ⏎  ⏎ Enabling shared-experts fusion (`--enforce-shared-experts-fusion`) for ⏎ DeepSeek-V4 on MI355X (gfx950) crashed at startup. Two independent issues in the ⏎ fused topk / append path assume DeepSeek-V3 conventions (fp32 correction bias, ⏎ power-of-two topk) that DeepSeek-V4 (bf16 correction bias, **top-6** routing) ⏎ violates. This PR fixes both so the fused path runs, and shows  …[truncated]

### L1-d287880a7a  (L1, 2026-08-19, sha d287880a7a76, PR #35450)
TITLE: Update deepep for SBO feature (#35450)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); test/registered/4-gpu-models/test_deepseek_v3_cutedsl_4gpu.py (+0/-4)
LABELS: dependencies, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L1-a5a9d66baf  (L1, 2026-08-20, sha a5a9d66bafa9, PR #34546)
TITLE: [XPU] Fix/kimi linear xpu (#34546)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+2/-1); python/sglang/srt/layers/attention/linear/kernels/kda_triton.py (+5/-2); python/sglang/srt/models/kimi_linear.py (+2/-2)
LABELS: intel, xpu, run-ci
BODY: ## Motivation ⏎ Enables KimiLinearForCausalLM (hybrid KDA linear-attention + MLA + MoE) to run on Intel XPU. ⏎  ⏎ ## Modifications ⏎  ⏎ - XPU has no tvm_ffi CUDA JIT kernel for KDA packed decode, so batched decode crashed. Set supports_packed_decode = ... and not is_xpu() so XPU uses the non-packed Triton decode() path (fused_sigmoid_gating_delta_rule_update) — the same fallback CPU/NPU already use ⏎ - grouped-topk kernels (sgl_kernel topk_sigmoid on X …[truncated]

### L1-92eeed41d7  (L1, 2026-08-20, sha 92eeed41d7b5, PR #35756)
TITLE: [Docker] Defer CUDA 13 NCCL override until after dependency resolution (#35756)
SOURCES: dependency_pin, body_keyword
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+12/-9)
BODY: ## Motivation ⏎  ⏎ The CUDA 13 development image build fails during the constrained installation of development packages and `runai-model-streamer`: ⏎ https://github.com/sgl-project/sglang/actions/runs/32400623762/job/96527778687 ⏎  ⏎ PyTorch 2.13 requires `nvidia-nccl-cu13==2.29.7`, but the Dockerfile previously force-installed 2.30.7 before generating `constraints.txt`. The resulting constraint conflicts with Torch metadata when pip resolves the later pa …[truncated]

### L1-7d893255c3  (L1, 2026-08-21, sha 7d893255c359, PR #34337)
TITLE: [Spec][LoRA] Support multi-adapter LoRA with EAGLE/NEXTN/DFLASH/DSPARK speculative decoding (#34337)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/lora/backend/base_backend.py (+6/-11); python/sglang/srt/lora/backend/chunked_backend.py (+2/-5); python/sglang/srt/lora/backend/triton_backend.py (+14/-11); python/sglang/srt/lora/layers.py (+5/-0); python/sglang/srt/lora/lora_manager.py (+23/-2); python/sglang/srt/lora/utils.py (+37/-0); python/sglang/srt/model_executor/forward_batch_info.py (+3/-3); python/sglang/srt/model_executor/model_runner.py (+5/-1); python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py (+1/-1); python/sglang/srt/model_executor/runner/base_runner.py (+1/-1); (+13 more)
LABELS: documentation, lora, run-ci
BODY: ## Motivation ⏎  ⏎ #12903 enabled LoRA with NGRAM speculative decoding. This extends it to **EAGLE / NEXTN / EAGLE3, DFLASH and DSPARK with multiple adapters co-batched**, one of the LoRA items in #11762. ⏎  ⏎ Adapters apply to the target model only; one shared draft runs unadapted. Speculation stays lossless per adapter (verify samples from the adapted target), so only the accept rate is affected. ⏎  ⏎ Supersedes #28395, which strips LoRA from a per-draft ` …[truncated]

### L1-834400705f  (L1, 2026-08-21, sha 834400705f2d, PR #34938)
TITLE: perf: overlap Qwen shared expert with DeepEP routed experts (#34938)
SOURCES: path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/qwen2_moe.py (+22/-0); docs/docs/references/environment_variables.mdx (+5/-0); python/sglang/srt/environ.py (+1/-0)
LABELS: documentation, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ For Qwen MoE models using CUDA and DeepEP, the shared expert currently runs on the default stream before the routed-expert path. This serializes two independent computations and leaves an available alternate stream unused. ⏎  ⏎ This PR adds an opt-in path that overlaps the Qwen shared expert with routed-expert/DeepEP execution. ⏎  ⏎ ## Modifications ⏎  ⏎ - Add `SGLANG_QWEN_DEEPEP_SHARED_OVERLAP` (default: `false`). ⏎ - When enabled on CUDA, enqu …[truncated]

### L1-60ff1e33a5  (L1, 2026-08-21, sha 60ff1e33a51f, PR #35919)
TITLE: [DeepSeek V4] Default FP4 checkpoints to FlashInfer MXFP4 MoE (#35919)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/arg_groups/overrides.py (+21/-24); python/sglang/srt/server_args.py (+0/-9); test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py (+2/-3); test/registered/unit/test_model_overrides.py (+83/-38)
LABELS: deepseek
BODY: ## Motivation ⏎  ⏎ DeepSeek V4 FP4 expert checkpoints require the FlashInfer MXFP4 MoE runner on supported NVIDIA GPUs. Select it automatically when the user leaves the MoE runner backend at `auto`. ⏎  ⏎ ## Modifications ⏎  ⏎ - Default DeepSeek V4 FP4 expert checkpoints to `flashinfer_mxfp4` on supported NVIDIA SM90, SM100, and SM120 devices. ⏎ - Preserve an explicitly selected `--moe-runner-backend`. ⏎ - Preserve the `nvidia/DeepSeek-V4-Pro-NVFP4` special case, …[truncated]

### L1-7fd5454335  (L1, 2026-08-21, sha 7fd5454335c1, PR #35175)
TITLE: [DSA] Route the ragged prefill top-k to the v2 kernel (#35175)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/dsv4/topk.py (+31/-2); python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh (+168/-6); python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh (+2/-1); python/sglang/srt/layers/attention/dsa/dsa_indexer_metadata.py (+1/-0); python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py (+50/-0); test/registered/kernels/benchmark/attention/bench_topk.py (+56/-5); test/registered/kernels/ops/attention/test_topk_v2.py (+112/-1)
LABELS: high priority, run-ci, jit-kernel, bypass-fastfail, run-ci-extra, release-highlight
BODY: > Generated by Claude. ⏎  ⏎ **Stacked on #35041.** GitHub cannot use that PR's branch as a base here (its head lives in the fork, and a cross-fork PR's base must be a branch of the base repo), so this targets `main` and carries #35041's commit underneath. **Review only the second commit** (`[DSA] Route the ragged prefill top-k to the v2 kernel`); the diff collapses to it once #35041 merges and this is rebased. ⏎  ⏎ ## Motivation ⏎  ⏎ The extend-shaped `RAGGE …[truncated]

### L1-3b5909de0e  (L1, 2026-08-21, sha 3b5909de0ef5, PR #35918)
TITLE: [DeepSeek V4] Add W4A4 MegaMoE server flag (#35918)
SOURCES: path_core, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.runner.deepgemm_megamoe
FILES: python/sglang/srt/arg_groups/mega_moe_hook.py (+38/-0); python/sglang/srt/layers/moe/mega_moe.py (+1/-23); .claude/skills/cookbook-add-model/templates/config.jsx.tmpl (+3/-6); docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx (+4/-5); docs/scripts/check_cookbook_configs.mjs (+16/-0); docs/src/snippets/_playground.jsx (+14/-9); docs/src/snippets/configs/Qwen/qwen3.8.jsx (+2/-5); docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx (+2/-5); docs/src/snippets/configs/moonshotai/kimi-k3.jsx (+2/-5); python/sglang/srt/environ.py (+6/-10); (+5 more)
LABELS: documentation, deepseek
BODY: ## Motivation ⏎  ⏎ DeepSeek V4 W4A4 MegaMoE currently requires users to set two SGLang-specific environment variables manually. Expose this mode through one server argument and configure DeepGEMM with its native environment variables. ⏎  ⏎ ## Modifications ⏎  ⏎ - Add `--enable-w4a4-megamoe`. ⏎ - Set `DG_USE_FP4_ACTS=1` and `DG_USE_MXF4_KIND=1` when the flag is enabled. ⏎ - Remove runtime handling of `SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS` and `SGLANG_OPT_DEE …[truncated]

### L1-af39ad9349  (L1, 2026-08-22, sha af39ad93493c, PR #33829)
TITLE: [Model] Complete dots.note.omni support with native encoders, video preprocessing, and MTP decoding (#33829)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: docs/cookbook/autoregressive/RedNote/Dots3-Note.mdx (+100/-20); docs/src/snippets/configs/rednote/dots3-note.jsx (+112/-60); python/sglang/srt/arg_groups/overrides.py (+3/-0); python/sglang/srt/configs/__init__.py (+2/-0); python/sglang/srt/configs/dots3.py (+243/-0); python/sglang/srt/configs/model_config.py (+24/-1); python/sglang/srt/entrypoints/openai/protocol.py (+1/-0); python/sglang/srt/entrypoints/openai/serving_chat.py (+36/-0); python/sglang/srt/function_call/dots_detector.py (+353/-0); python/sglang/srt/function_call/function_call_parser.py (+2/-0); (+45 more)
LABELS: documentation, high priority, Multi-modal, deepseek, run-ci, jit-kernel, bypass-fastfail, run-ci-extra, release-highlight, memory-pool
BODY: ## Motivation ⏎  ⏎ Merge dots.note.omni model ⏎  ⏎ ## Modifications ⏎  ⏎ This PR completes the SGLang integration of dots.note.omni, including: ⏎  ⏎   - Native in-process vision and audio encoders ⏎   - Train-consistent native video preprocessing ⏎   - Full-sharing MTP/NextN speculative decoding ⏎   - DP/TP/EP execution support, including overlap scheduling ⏎   - Correct KV-cache sizing for the hybrid sliding-window draft model ⏎  ⏎ ## How dots.note.omni diffe …[truncated]

### L1-b98d472158  (L1, 2026-08-22, sha b98d472158f7, PR #34855)
TITLE: [NPU] [Diffusion] Fix critical Ascend NPU Diffusion regression/bugs & restore 2-NPU CI testcase (#34855)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/npu.Dockerfile (+1/-1); .github/workflows/diffusion-ci-gt-gen-npu.yml (+6/-2); .github/workflows/pr-test-npu.yml (+20/-6); .github/workflows/release-docker-npu-nightly.yml (+1/-1); .github/workflows/release-docker-npu.yml (+1/-1); python/sglang/kernels/ops/attention/flash_attention.py (+0/-1); python/sglang/multimodal_gen/runtime/layers/attention/backends/ascend_fa.py (+205/-1); python/sglang/multimodal_gen/runtime/layers/attention/backends/attention_backend.py (+15/-0); python/sglang/multimodal_gen/runtime/layers/attention/backends/flash_attn.py (+34/-0); python/sglang/multimodal_gen/runtime/layers/attention/layer.py (+1/-1); (+10 more)
LABELS: npu, run-ci, diffusion, jit-kernel
BODY: ## Motivation ⏎  ⏎ Restore Ascend/NPU packed Ring Attention, 2-NPU diffusion CI suite, fix NPU correctness, runtime, and CI regressions exposed by recent Ring, SRT-CLIP, residency, and MOVA changes. ⏎  ⏎ After [#35004](https://github.com/sgl-project/sglang/pull/35004), diffusion can import SRT CLIP code and execute SRT `init_npu_backend()`. Importing `torch_npu.contrib.transfer_to_npu` inside a native NPU diffusion process globally rewrites CUDA-faci …[truncated]

### L1-edd675cecf  (L1, 2026-08-22, sha edd675cecfb2, PR #34490)
TITLE: [AMD] Add Radix-4 MoE top-k router kernel for Kimi-K3 routing (#34490)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/kernels/jit/csrc/moe/route_radix4_hip.cuh (+527/-0); python/sglang/kernels/ops/moe/moe_route_radix4.py (+137/-0); python/sglang/srt/layers/moe/topk.py (+18/-2); python/sglang/srt/environ.py (+2/-0); test/registered/kernels/ops/moe/test_moe_route_radix4.py (+307/-0)
LABELS: amd, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: co-author: @kkHuang-amd  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎ Kimi-K3 routes 16 of 896 experts, ungrouped. That lands on aiter's generic `biased_grouped_topk`, which spends one round per selected expert — cost tracks `topk`, ~10.4us per layer on MI355X. aiter's faster pivot-based path is gated on DeepSeek's exact shape (256 experts, 8 groups, top-8), so K3 never reaches it. ⏎  ⏎ Set `SGLANG_K3_RADIX4_TOPK=1` to enable the radix4 topk path. ⏎  ⏎ ## Modifications ⏎  ⏎  …[truncated]

### L1-0e22777572  (L1, 2026-08-23, sha 0e22777572eb, PR #35905)
TITLE: config: record resolution writes in a declaration stash (#35905)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/arg_groups/mega_moe_hook.py (+8/-2); python/sglang/srt/arg_groups/arg_utils.py (+8/-0); python/sglang/srt/arg_groups/deepseek_v4_hook.py (+36/-7); python/sglang/srt/arg_groups/kimi_k3_hook.py (+22/-4); python/sglang/srt/arg_groups/overrides.py (+45/-4); python/sglang/srt/arg_groups/pd_disaggregation_hook.py (+31/-6); python/sglang/srt/arg_groups/speculative_hook.py (+210/-51); python/sglang/srt/hardware_backend/npu/utils.py (+51/-10); python/sglang/srt/server_args.py (+601/-183); test/registered/unit/server_args/test_resolution_declarations.py (+381/-0); (+2 more)
LABELS: deepseek, speculative-decoding, ready-to-merge
BODY: ## Stack ⏎  ⏎ Part of a series that moves `ServerArgs` from "mutate the record at construction" to "resolve once, publish into config bags". Each PR stands alone (builds, passes its own tests); review bottom-up. ⏎  ⏎ | # | PR | base | ⏎ |---|----|------| ⏎ | 0 | #35904 | `main` | ⏎ | 1 | #35905 | #35904 | ⏎ | 2 | #35906 | #35905 | ⏎ | 3 | #35907 | #35906 | ⏎ | 4 | #35908 | #35907 | ⏎ | 5 | #35909 | #35908 | ⏎ | 6 | #35910 | #35909 | ⏎  ⏎ Whole-series CI vehicle (not for mer …[truncated]

### L1-362c2ee849  (L1, 2026-08-23, sha 362c2ee849cf, PR #35908)
TITLE: config: borrowed-record reads follow the config bags (#35908)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.runner.marlin
FILES: python/sglang/srt/layers/moe/kt_ep_wrapper.py (+5/-2); python/sglang/benchmark/one_batch.py (+6/-6); python/sglang/srt/disaggregation/common/staging_handler.py (+5/-1); python/sglang/srt/disaggregation/decode.py (+11/-11); python/sglang/srt/disaggregation/prefill.py (+2/-3); python/sglang/srt/disaggregation/utils.py (+11/-15); python/sglang/srt/distributed/bootstrap.py (+10/-7); python/sglang/srt/entrypoints/grpc_bridge.py (+6/-4); python/sglang/srt/entrypoints/http_server.py (+5/-4); python/sglang/srt/entrypoints/sidecar.py (+8/-7); (+55 more)
LABELS: quant, Multi-modal, deepseek, speculative-decoding, ready-to-merge, diffusion, model-gateway, apple-silicon
BODY: ## Stack ⏎  ⏎ Part of a series that moves `ServerArgs` from "mutate the record at construction" to "resolve once, publish into config bags". Each PR stands alone (builds, passes its own tests); review bottom-up. ⏎  ⏎ | # | PR | base | ⏎ |---|----|------| ⏎ | 0 | #35904 | `main` | ⏎ | 1 | #35905 | #35904 | ⏎ | 2 | #35906 | #35905 | ⏎ | 3 | #35907 | #35906 | ⏎ | 4 | #35908 | #35907 | ⏎ | 5 | #35909 | #35908 | ⏎ | 6 | #35910 | #35909 | ⏎  ⏎ Whole-series CI vehicle (not for mer …[truncated]

### L1-a90d770c40  (L1, 2026-08-23, sha a90d770c40c3, PR #33684)
TITLE: [Weight Cache] Support static DP/EP layouts (#33684)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/mooncake.py (+8/-2); python/sglang/srt/entrypoints/engine.py (+15/-57); python/sglang/srt/server_args.py (+5/-0); python/sglang/srt/weight_cache/daemon.py (+202/-280); python/sglang/srt/weight_cache/ipc_loader.py (+13/-2); python/sglang/srt/weight_cache/protocol.py (+8/-0); test/manual/test_weight_cache_e2e.py (+8/-4); test/registered/unit/model_loader/test_weight_cache_protocol.py (+77/-1); test/registered/unit/test_supplied_instance_exposure_ratchet.py (+21/-0)
LABELS: run-ci, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Weight-cache daemons and engines must construct identical model-parallel layouts before CUDA IPC can safely map a cached MoE shard. The existing daemon launcher rejected `dp_size > 1`, preventing static DP/EP deployments from using the daemon-backed weight cache. ⏎  ⏎ This change adds static DP/EP layout support to the CUDA IPC weight-cache path. It is the implementation foundation for the DP/EP roadmap in #33522. ⏎  ⏎ ## Modifications ⏎  ⏎ -  …[truncated]

### L1-b498efce52  (L1, 2026-08-23, sha b498efce5209, PR #36053)
TITLE: chore: move cuda_vmm_utils.py under srt/utils/ (#36053)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/dwdp/layout.py (+1/-1); python/sglang/srt/layers/moe/dwdp/page_pool.py (+1/-1); python/sglang/srt/layers/moe/dwdp/transport.py (+6/-6); python/sglang/srt/layers/moe/dwdp/weight_buffer.py (+6/-6); python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py (+1/-1); python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py (+5/-5); python/sglang/srt/mem_cache/kv_vmm_backing.py (+1/-1); python/sglang/srt/model_executor/runner_utils/pool.py (+1/-1); python/sglang/srt/multimodal/transport/memory_pool.py (+1/-1); python/sglang/srt/utils/cuda_vmm_transport_utils.py (+16/-16); (+2 more)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ `python/sglang/srt/cuda_vmm_utils.py` sat at the top level of `srt/`, while its closest relatives — `cuda_vmm_transport_utils.py` and `cuda_ipc_transport_utils.py` — already live under `srt/utils/`. Move it next to them. ⏎  ⏎ ## Modifications ⏎  ⏎ - `git mv python/sglang/srt/cuda_vmm_utils.py python/sglang/srt/utils/cuda_vmm_utils.py` — the file contents are **unchanged** (rename shows 100% similarity). ⏎ - Updated every import site from `sg …[truncated]

### L1-7bbd0ddeb5  (L1, 2026-08-24, sha 7bbd0ddeb5f3, PR #36124)
TITLE: [AMD] Quark shared-experts gate: recognise a trailing MTP layer (#36124)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/quark/quark.py (+35/-1)
LABELS: amd, run-ci
BODY: ## Summary ⏎  ⏎ `can_fuse_shared_expert()` recognises the MTP draft stack only by an `mtp.` prefix. GLM-5.2-MXFP4 spells it `model.layers.78` (`num_hidden_layers=78`), so three excluded draft projections veto fusion for all 78 target-model layers. `from_config()`'s prequantized branch also never forwarded `hf_config`, leaving the layer count unavailable. ⏎  ⏎ Losing fusion costs more than the fused shared expert alone: ⏎  ⏎ |                      | fus …[truncated]

### L1-56834422a1  (L1, 2026-08-24, sha 56834422a18a, PR #33323)
TITLE: [Intel XPU] Add xpu pass for biased_topk and hash_topk (#33323)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py, L1.routing.hash_topk
FILES: python/sglang/srt/layers/moe/hash_topk.py (+37/-2); python/sglang/srt/layers/moe/topk.py (+43/-1); test/registered/xpu/test_topk.py (+153/-6)
LABELS: intel, xpu, run-ci
BODY: ## Modifications ⏎ Enable sycl pass of biased_topk and hash_topk ⏎  ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #32334621079](https://github.com/sgl-project/sglang/actions/runs/32334621079) ⏎ Latest PR Test (Extra): :x: [Run #32334620929](https://github.com/sgl-project/sglang/actions/runs/32334620929) ⏎ Latest PR Test (AMD ROCm 7.2): :x: [Run #32334620961](https://github.com/sgl-project/sglang/actions/runs/32334620961)

### L1-f98b60de80  (L1, 2026-08-24, sha f98b60de806d, PR #33057)
TITLE: fix(xpu): enable compressed-tensors FP8 W8A8 on XPU (RedHatAI FP8-dynamic models) (#33057)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/quantization/fp8_kernel.py (+4/-0); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py (+23/-5); python/sglang/srt/layers/quantization/fp8.py (+4/-1); python/sglang/srt/layers/quantization/fp8_utils.py (+5/-1); python/sglang/srt/models/internvl.py (+3/-1); python/sglang/srt/runtime_context.py (+3/-2)
LABELS: quant, intel, xpu, run-ci, jit-kernel, run-ci-extra
BODY: ## Motivation ⏎  ⏎ compressed-tensors FP8 W8A8 quantized models (e.g. RedHatAI's `*-FP8-dynamic` family — `Apertus-8B-Instruct-2509-FP8-dynamic`, `Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic`, `granite-4.0-h-small-FP8-dynamic`, `NVIDIA-Nemotron-Nano-9B-v2-FP8-dynamic`) currently fail to serve on XPU. The very first failure happens at layer-construction time, before any kernel dispatch: ⏎  ⏎ ``` ⏎ AssertionError: Torch not compiled with CUDA enable …[truncated]

### L1-0665740953  (L1, 2026-08-24, sha 06657409533d, PR #32039)
TITLE: [AMD][Fix] Route MoRI through the Qwen MoE all-to-all path (#32039)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/qwen2_moe.py (+4/-2); test/registered/unit/models/test_shared_experts_fusion_gates.py (+39/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Serving **Qwen3.5 MoE** (Qwen3.5-397B-A17B-MXFP4) with `--moe-a2a-backend mori` on MI355X/ROCm (DP=1 / TP=2, EP=2) produced near-zero accuracy — gsm8k **0.006**. ⏎  ⏎ The Qwen2/Qwen3 MoE gates key on `get_moe_a2a_backend().is_deepep()`, which recognises **only** pure DeepEP. MoRI is a DeepEP-class backend and needs the same per-rank EP expert layout, so under MoRI the block silently fell through to the plain-TP path: ⏎  ⏎ - the shared expe …[truncated]

### L1-0d5b5ae620  (L1, 2026-08-24, sha 0d5b5ae6202c, PR #35719)
TITLE: [AMD] Fix Qwen3.5 MTP dropping fused shared-expert weights (#35719)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/qwen3_5_mtp.py (+24/-2)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (perf_regression_fix)
BODY: ## Motivation ⏎  ⏎ Since #33889, Qwen3.5 MXFP4 with MTP speculative decoding on MI355X lost about 12% throughput. The cause is that the MTP draft model is loaded without its shared-expert weights. ⏎  ⏎ **What fusion does to the weight layout.** When a Qwen3.5 MoE layer fuses its shared expert, it deliberately does not build a separate `shared_expert` module — the shared expert becomes one more routed expert, living in slot `num_experts`. Any loader for s …[truncated]

### L1-77940dec80  (L1, 2026-08-24, sha 77940dec80c9, PR #34915)
TITLE: [MoE] Gather the cutlass MoE activation and its scales in one launch (#34915)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: L1.cutlass.adapters
FILES: python/sglang/kernels/ops/moe/__init__.py (+13/-0); python/sglang/kernels/ops/moe/shuffle_rows_with_scales.py (+137/-0); python/sglang/srt/layers/moe/cutlass_moe.py (+8/-2); test/registered/kernels/benchmark/moe/bench_shuffle_rows_with_scales.py (+69/-0); test/registered/kernels/ops/moe/test_shuffle_rows_with_scales.py (+123/-0)
LABELS: run-ci, jit-kernel, run-ci-extra
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ The fp8 blockwise CUTLASS MoE quantizes its activation once and then replicates rows per routed expert, so the gather it performs walks a `dst2src` map of `m * topk` entries. Until now it walked that map twice: one `shuffle_rows` launch for the fp8 values and a second one for the fp32 group scales. ⏎  ⏎ The second walk moves a thirty-second of the bytes the first one does — `k // 128` fp32 against `k` fp8 — so as a launch of its ow …[truncated]

### L1-91e7e84ee5  (L1, 2026-08-24, sha 91e7e84ee5a0, PR #35116)
TITLE: [SM120] flash_mla: allocate the page-split buffer outside inference mode (#35116)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/flash_mla_sm120.py (+10/-6)
LABELS: jit-kernel
BODY: ## Purpose ⏎  ⏎ `_split_kv_pages_to_64` keeps two lazily allocated persistent buffers in the same ⏎ `buffers` dict, with the same lifetime: allocated once on first use, reused across ⏎ autotune, CUDA graph capture and steady-state serving. ⏎  ⏎ On current main only one of them is protected: ⏎  ⏎ ```python ⏎     buf = buffers.get(key)                      # page-split destination ⏎     if buf is None or buf.shape[0] < num_dst_pages: ⏎         buf = torch.empty(...)     …[truncated]

### L1-21258b7a35  (L1, 2026-08-24, sha 21258b7a35e9, PR #31429)
TITLE: feat(humming): FP8 DeepEP dispatch for humming MoE backend (#31429)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.ep.layer, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/kernels/ops/moe/ep_moe_kernels.py (+50/-13); python/sglang/srt/layers/moe/moe_runner/humming.py (+162/-19); python/sglang/srt/layers/quantization/humming.py (+12/-3); python/sglang/srt/layers/quantization/humming_utils.py (+53/-2); test/registered/kernels/ops/moe/test_humming_fp8_dispatch.py (+265/-0)
LABELS: dependencies, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Summary ⏎  ⏎ FP8 DeepEP dispatch for the Humming MoE backend: adds `moe_permute_with_scale` ⏎ (row-permutes hidden states together with their group-128 FP8 scales), dispatch ⏎ input validation, and a fused masked activation+quant fast path. Bumps ⏎ `humming-kernels` to 0.1.12. ⏎  ⏎ ## How to run ⏎  ⏎ Server (2 nodes × 8 H20, DeepSeek-V4-Pro pp2/tp8/ep8; run on each node with ⏎ `--node-rank 0/1`): ⏎  ⏎ ```bash ⏎ # fp8 arm only: ⏎ export SGLANG_HUMMING_INPUT_QUANT_CONFIG=' …[truncated]

### L1-bf1e03f712  (L1, 2026-08-25, sha bf1e03f7123e, PR #36237)
TITLE: [MegaMoE] Respect padded MXFP8 scale row strides in pre-dispatch (#36237)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/jit/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh (+16/-9); test/registered/kernels/ops/moe/test_mega_moe_pre_dispatch.py (+70/-0)
LABELS: run-ci, jit-kernel
BODY: Replaces #36007. This branch was rebuilt directly on the latest required `main` commit (`1ec20fd`) because the prior branch was blocked by the CI rebase gate. ⏎  ⏎ ## Summary ⏎  ⏎ Update `mega_moe_pre_dispatch` to honor the physical row stride of DeepGEMM's MXFP8 scale output instead of assuming packed rows. ⏎  ⏎ DeepGEMM stores one scale byte per 32 hidden elements and aligns each token row to 16 bytes for TMA: ⏎  ⏎     logical row bytes  = H / 32 ⏎     physical  …[truncated]

### L1-2e3934f4cb  (L1, 2026-08-25, sha 2e3934f4cba6, PR #35630)
TITLE: [AMD] Enable Mori-EP on kimi-k3 (#35630)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/mxfp4.py (+77/-58); python/sglang/srt/models/kimi_k3.py (+8/-6)
LABELS: amd, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ Enable Mori EP for Kimi-K3 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ **`models/kimi_k3.py`**:  ⏎ add `is_mori()` to the `_ep_a2a` and `_sp_moe` predicates so the model recognises mori as an EP-a2a backend. ⏎  ⏎ **`layers/quantization/mxfp4.py`**:  ⏎ `apply()` unpacks `.topk_output`, which DeepEP-family dispatch outputs (mori included) do not have. Return early to the ⏎ AITER runner before that unpack, the same way the `use_deep_gemm` branch above al …[truncated]

### L1-d2d8ecea77  (L1, 2026-08-25, sha d2d8ecea778a, PR #36180)
TITLE: [npu] Combine NPU test fixes from #35472 and #34516 (#36180)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test-npu.yml (+4/-4); python/sglang/test/ascend/e2e/test_npu_accuracy_utils.py (+4/-19); python/sglang/test/ascend/e2e/test_npu_multi_node_utils.py (+20/-0); python/sglang/test/ascend/e2e/test_npu_performance_utils.py (+57/-0); test/registered/npu/accuracy/qwen3_vl_30b_a3b_thinking/test_npu_qwen3_vl_30b_a3b_thinking_1p_mmmu.py (+2/-2); test/registered/npu/accuracy/qwen3_vl_8b_thinking/test_npu_qwen3_vl_8b_thinking_1p_mmmu.py (+3/-3); test/registered/npu/performance/kimi_k2_6/test_npu_kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms.py (+2/-0)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  ⏎ Improve reliability of Ascend NPU benchmark and accuracy tests by detecting hung benchmark processes earlier and cleaning up orphaned process groups, and align several NPU test configurations and test class naming. ⏎  ⏎ ## Modifications ⏎  ⏎ - Add `kill_process_group` and use it in the NPU accuracy/benchmark helpers to reap re-parented deep-ep/HCCL worker processes. ⏎ - Add `BENCHMARK_STDOUT_IDLE_TIMEOUT` and rework the benchmark watchdog to …[truncated]

### L1-5bcc978f2a  (L1, 2026-08-25, sha 5bcc978f2aaf, PR #36306)
TITLE: [AMD][CI] Pass USE_PDL explicitly in the fused MoE gate (#36306)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/elementwise/elementwise.py (+3/-1)
LABELS: run-ci, jit-kernel
ISSUES: #35348 [Bug] [AMD] torch.compile: "launcher() missing 1 required positional argument: '_grid_2'" in _fused_gate_sigmoid_mul_add_kernel on ROCm + torch 2.11 / Triton 3.7
BODY: ## Motivation ⏎  ⏎ Fixes #35348, `test_torch_compile_moe.py` in [stage-b-test-1-gpu-small-amd-rocm720 (linux-mi300-1gpu-sglang, 0)](https://github.com/sgl-project/sglang/actions/runs/32367546070/job/96420375810#logs) ⏎  ⏎ ROCm 7.2.4 with torch 2.11 and AMD Triton 3.7 fails while capturing Qwen1.5-MoE with torch.compile: ⏎  ⏎     TypeError: launcher() missing 1 required positional argument: '_grid_2' ⏎  ⏎ On HIP, PDL is unsupported, so the fused gate laun …[truncated]

### L1-a1f9508dd4  (L1, 2026-08-25, sha a1f9508dd410, PR #35188)
TITLE: [Bugfix] Fix int32 destination offset overflow in CUTLASS MoE pre-reorder (#35188)
SOURCES: path_core, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/kernels/ops/moe/ep_moe_kernels.py (+1/-1)
LABELS: jit-kernel
DEEP_STUDY: deep-study correctness case sglang:a1f9508dd4: class=memory_safety_oob; symptom=illegal_memory_access; introducing=unknown
BODY: ## Motivation ⏎  ⏎ <img width="755" height="561" alt="image" src="https://github.com/user-attachments/assets/33953193-ec5e-458b-bcd3-971e9643db58" /> ⏎  ⏎ `pre_reorder_triton_kernel_for_cutlass_moe` loads `dst_idx` from the `src2dst` mapping as an `int32` and uses it to calculate the destination address: ⏎  ⏎ ```python ⏎ dst_ptr_offs + dst_idx * hidden_size ⏎ ``` ⏎  ⏎ Because `dst_idx` is 32-bit, `dst_idx * hidden_size` can overflow signed 32-bit arithmeti …[truncated]

### L1-8005df61d3  (L1, 2026-08-26, sha 8005df61d32c, PR #36250)
TITLE: config: spell the parallel config tier at the call site (#36250)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.flashinfer_cutedsl, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-2); python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py (+1/-1); python/sglang/srt/layers/moe/token_dispatcher/nixl.py (+2/-2); python/sglang/srt/layers/moe/token_dispatcher/pplx.py (+1/-1); python/sglang/srt/layers/moe/utils.py (+1/-1); .claude/skills/sglang-runtime-context/SKILL.md (+59/-46); python/sglang/benchmark/one_batch.py (+1/-1); python/sglang/srt/batch_overlap/two_batch_overlap.py (+1/-1); python/sglang/srt/disaggregation/common/conn.py (+17/-14); python/sglang/srt/disaggregation/encoder/http_server.py (+1/-1); (+126 more)
LABELS: documentation, amd, lora, deepseek, hicache, ready-to-merge
BODY: ## Motivation ⏎  ⏎ `get_parallel()` exposes two different things through one spelling. Bare ⏎ `get_parallel().tp_size` answers from the **live** process groups; the ⏎ `configured_*_size()` accessors answered from the **published** `parallel` config. Both ⏎ spellings work everywhere, neither says which one it is, and they are not ⏎ interchangeable — they provably differ in three situations: ⏎  ⏎ 1. before `torch.distributed` is initialised, or in a process that h …[truncated]

### L1-5b7fc61306  (L1, 2026-08-26, sha 5b7fc6130613, PR #36253)
TITLE: config: resolution reads the declarations, not the fields (#36253)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.runner.marlin
FILES: python/sglang/srt/arg_groups/mega_moe_hook.py (+10/-5); python/sglang/srt/arg_groups/deepseek_v4_hook.py (+38/-38); python/sglang/srt/arg_groups/expert_pack_hook.py (+15/-14); python/sglang/srt/arg_groups/hisparse_hook.py (+5/-2); python/sglang/srt/arg_groups/kimi_k3_hook.py (+16/-10); python/sglang/srt/arg_groups/overrides.py (+179/-135); python/sglang/srt/arg_groups/pd_disaggregation_hook.py (+35/-32); python/sglang/srt/arg_groups/speculative_hook.py (+151/-147); python/sglang/srt/configs/model_config.py (+26/-26); python/sglang/srt/dllm/config.py (+11/-10); (+24 more)
LABELS: Multi-modal, deepseek, speculative-decoding, ready-to-merge, npu
BODY: ## Motivation ⏎  ⏎ Resolution is a chain: one resolver decides a field, the next one reads that decision. ⏎ Today that works only because `declare_resolution` writes the field as a side effect, so ⏎ the record doubles as the scratchpad for a half-finished resolution. That side effect is ⏎ what keeps `ServerArgs` from being what it should be — the raw user input — and it makes ⏎ "who decided this value" unanswerable after the fact. ⏎  ⏎ This PR moves every read t …[truncated]

### L1-937af8538b  (L1, 2026-08-26, sha 937af8538b07, PR #36254)
TITLE: config: the runtime readers take the published bags (#36254)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/utils.py (+34/-21); .claude/skills/sglang-runtime-context/SKILL.md (+10/-10); python/sglang/benchmark/offline_throughput.py (+9/-6); python/sglang/benchmark/one_batch.py (+69/-38); python/sglang/benchmark/one_batch_server.py (+3/-1); python/sglang/compile_deep_gemm.py (+28/-13); python/sglang/lang/backend/runtime_endpoint.py (+6/-4); python/sglang/launch_server.py (+6/-4); python/sglang/srt/configs/embedding_model_spec.py (+8/-12); python/sglang/srt/disaggregation/encoder/grpc_server.py (+11/-4); (+57 more)
LABELS: documentation, hicache, ready-to-merge, unified-radix-cache
BODY: ## Motivation ⏎  ⏎ These are the readers that run after the config is published and still read the record. ⏎ Each one is a place where a value the user did not set — one that resolution decided — is ⏎ read from the wrong side. Two of them were user-visible: ⏎  ⏎ - The Ray driver sized its placement groups from a mix of both surfaces; one line read ⏎   `get_parallel().config.pp_size * server_args.tp_size`. A launch that lets resolution ⏎   decide `dp_size` comput …[truncated]

### L1-413df1f8db  (L1, 2026-08-26, sha 413df1f8db4f, PR #36255)
TITLE: config: ServerArgs holds the raw input (#36255)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/utils.py (+2/-2); .claude/skills/sglang-runtime-context/SKILL.md (+54/-25); examples/runtime/engine/save_remote_state.py (+2/-1); examples/runtime/engine/save_sharded_state.py (+2/-1); examples/runtime/token_in_token_out/token_in_token_out_vlm_engine.py (+5/-3); python/sglang/benchmark/one_batch.py (+1/-1); python/sglang/srt/arg_groups/arg_utils.py (+4/-4); python/sglang/srt/arg_groups/overrides.py (+43/-62); python/sglang/srt/entrypoints/engine.py (+2/-2); python/sglang/srt/entrypoints/http_server.py (+2/-2); (+24 more)
LABELS: documentation, Multi-modal, speculative-decoding, ready-to-merge, npu, model-gateway
BODY: ## Motivation ⏎  ⏎ Everything up to here made the record's fields unnecessary as a communication channel: ⏎ resolution reads the declaration stash, and the runtime readers read the published bags. ⏎ What is left is the side effect itself — `declare_resolution` still writes the field it ⏎ declares. While it does, `ServerArgs` is neither the user's input nor the resolved ⏎ configuration but a mutable mixture of both, and no code can ask "what did the user ⏎ actu …[truncated]

### L1-2d8484740d  (L1, 2026-08-26, sha 2d8484740d5e, PR #35314)
TITLE: Support deepseek v4 and kimi k3 on ssd (#35314)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/jit/csrc/moe/expert_pack_mxfp4.cu (+565/-0); python/sglang/kernels/ops/moe/expert_pack_mxfp4.py (+113/-0); python/sglang/srt/layers/moe/expert_pack.py (+931/-0); python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py (+2/-0); examples/runtime/deepseek_v4/benchmark_deepseek_5090.py (+492/-0); examples/runtime/kimi_k3/benchmark_kimi_k3_5090.py (+573/-0); python/sglang/kernels/jit/csrc/ngram_corpus/result.h (+1/-0); python/sglang/kernels/ops/attention/dsv4/compress.py (+7/-0); python/sglang/kernels/ops/kimi_k3/attn_res.py (+4/-3); python/sglang/srt/arg_groups/expert_pack_hook.py (+198/-0); (+36 more)
LABELS: quant, deepseek, run-ci, jit-kernel, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: # [Feature] Add SSD-backed Expert Pack inference for DeepSeek-V4-Flash and Kimi-K3 ⏎  ⏎ ## Summary ⏎  ⏎ This PR adds an opt-in SSD-backed inference path for **DeepSeek-V4-Flash** and **Kimi-K3**. It allows these MoE models to run when their complete weights are larger than GPU VRAM and host RAM combined. ⏎  ⏎ Routed expert weights are stored on SSD and loaded only when selected by the router. The implementation uses: ⏎  ⏎ - an expert-major **Expert Pack* …[truncated]

### L1-cc3b61873f  (L1, 2026-08-26, sha cc3b61873f5c, PR #35850)
TITLE: [Diffusion][minimax-h3] Restrict MiniMax-H3 SubBlock sparsity to video queries (#35850)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/router.py (+3/-2); python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py (+40/-11); python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse_attn.py (+115/-31); python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py (+68/-9); python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py (+79/-1); python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/packed_sequence.py (+21/-5); python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/presentation.py (+48/-16); python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py (+24/-1); python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/text_encoding.py (+44/-12); python/sglang/multimodal_gen/test/unit/test_minimax_h3_admission.py (+63/-11); (+3 more)
LABELS: run-ci, diffusion, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎ MiniMax-H3 packs text, condition-image/video, target-video, and audio tokens into the same denoising sequence. The existing SubBlock path applies sparse attention to every query in that packed sequence, even though the sparsity policy is primarily intended for video queries. ⏎  ⏎ The audio latent sequence is substantially shorter than the video latent sequence and does not have the same spatial redundancy as video tokens. Applying  …[truncated]

### L1-27c36368b6  (L1, 2026-08-26, sha 27c36368b638, PR #36275)
TITLE: fix(moe): guard FP8 delegate activation params (#36275)
SOURCES: path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/fp8.py (+8/-1); test/registered/unit/layers/quantization/test_fp8_moe_runner_ownership.py (+118/-0)
LABELS: run-ci, bypass-fastfail
ISSUES: #36264 [Bug] AttributeError: 'Fp8MoEMethod' object has no attribute 'moe_runner_config' when using flashinfer_trtllm_routed MoE backend with speculative decoding (DSPARK/EAGLE) on hybrid NVFP4 checkpoints
BODY: ## Motivation ⏎  ⏎ Fixes #36264. When `Fp8MoEMethod` is used as a delegate by the hybrid NVFP4 FlashInfer TRT-LLM MoE method, it does not construct its own `MoeRunner` or receive a `moe_runner_config`. The post-load path nevertheless tries to materialize TRT-LLM activation parameters and raises an `AttributeError`. ⏎  ⏎ ## Modifications ⏎  ⏎ - Track whether `Fp8MoEMethod` owns a constructed `MoeRunner`. ⏎ - Materialize FlashInfer TRT-LLM activation parameters  …[truncated]

### L1-2935bb8e79  (L1, 2026-08-26, sha 2935bb8e79e6, PR #36456)
TITLE: Fix OOB read in mxfp4 MoE weight scales on Hopper (#36456)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/mxfp4.py (+12/-0)
LABELS: high priority, run-ci, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎ `test_gpt_oss_4gpu_mxfp4.py` is intermittently red on `4-gpu-h100`. It is not an ⏎ accuracy failure -- the server dies mid-eval and every request returns empty, so the ⏎ score is exactly `0.0`: ⏎  ⏎ ``` ⏎ coredump: Detected an exception of type CUDBG_EXCEPTION_WARP_ILLEGAL_ADDRESS (14) ⏎ coredump:   #0  _matmul_NNT_bf16xbf16xmxfp4_32x256x128x1_swiglu   (_matmul.py:371) ⏎ AssertionError: 0.0 not greater than or equal to 0.58 ⏎ ``` ⏎  ⏎ ## R …[truncated]

### L1-3ce243da3f  (L1, 2026-08-26, sha 3ce243da3f3d, PR #36233)
TITLE: [NVIDIA] Add CUDA 13.4 container for initial Rubin support (#36233)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.cu134 (+1167/-0); python/sglang/kernels/aot/CMakeLists.txt (+5/-4); python/sglang/kernels/aot/cmake/flashmla.cmake (+2/-2); .github/workflows/release-docker-cu134-nightly.yml (+104/-0)
LABELS: sgl-kernel, nvidia
BODY: ## Motivation ⏎  ⏎ [CUDA 13.4 Developer Preview](https://docs.nvidia.com/cuda/developer-preview/13.4/index.html) provides initial support for Rubin hardware (sm_107). This PR adds a new docker image with CUDA 13.4 for functional Rubin support, which would be built alongside the current cu12 and cu13 containers. ⏎  ⏎ ## Modifications ⏎  ⏎ Adds `docker/Dockerfile.cu134` which is based on the standard sglang container Dockerfile, with a few key changes. U …[truncated]

### L1-20621aa14b  (L1, 2026-08-26, sha 20621aa14bda, PR #33561)
TITLE: [Model] Support Ling-3.0-flash (BailingMoeV3)  (#33561)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels, L1.routing.topk_py, L1.runner.flashinfer_cutlass, L1.routing.router_py
FILES: python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py (+25/-1); python/sglang/kernels/ops/moe/router.py (+125/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+24/-17); python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py (+9/-1); python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+54/-27); python/sglang/srt/layers/moe/topk.py (+13/-3); python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py (+16/-2); .claude/skills/sglang-runtime-context/SKILL.md (+2/-1); benchmark/kernels/fused_moe_triton/common_utils.py (+1/-0); python/sglang/kernels/aot/csrc/allreduce/custom_all_reduce.cuh (+3/-0); (+66 more)
LABELS: documentation, quant, amd, deepseek, sgl-kernel, blackwell, npu, run-ci, jit-kernel, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Day-0 support for [inclusionAI/Ling-3.0-flash](https://huggingface.co/inclusionAI/Ling-3.0-flash) (`BailingMoeV3ForCausalLM`, `model_type=bailing_hybrid`): a hybrid MoE architecture interleaving KDA linear attention (with a safe-gate lower bound) and MLA, with MTP (NEXTN) and DSPARK speculative decoding support. ⏎  ⏎ ## Modifications ⏎  ⏎ - `BailingMoeV3ForCausalLM` model (`bailing_hybrid` config): KDA + MLA hybrid layers, 512-expert MoE,  …[truncated]

### L1-4f59a8dcfa  (L1, 2026-08-26, sha 4f59a8dcfa04, PR #36309)
TITLE: [AMD][Bugfix] Skip invalid fused MoE reduction for direct top-1 output (#36309)
SOURCES: path_core, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+3/-1)
BODY: ## Motivation ⏎  ⏎ ROCm 7.2.4 [stage-b-test-1-gpu-small-amd (rocm724, linux-mi300-1gpu-sglang, 7)](https://github.com/sgl-project/sglang/actions/runs/32725797196/job/97426719871#logs) fails `TestFusedMOE.test_single_expert_routing` for `topk=1` and `routed_scaling_factor=1.0`. ⏎  ⏎ In this path, the second GEMM writes directly to `out_hidden_states`, so `intermediate_cache3` is not populated. The HIP epilogue nevertheless reduces `intermediate_cache3 …[truncated]

### L1-a3ae667d67  (L1, 2026-08-26, sha a3ae667d67a4, PR #35634)
TITLE: [Feature] Add DeepEPv2 (ElasticBuffer) MoE A2A backend  (#35634)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.framework, L1.runner.deep_gemm, L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/kernels/ops/moe/ep_moe_kernels.py (+363/-0); python/sglang/srt/layers/moe/ep_moe/layer.py (+4/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+39/-0); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+192/-1); python/sglang/srt/layers/moe/moe_runner/runner.py (+9/-0); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+8/-0); python/sglang/srt/layers/moe/token_dispatcher/base.py (+19/-0); python/sglang/srt/layers/moe/token_dispatcher/deepep_v2.py (+460/-0); python/sglang/srt/layers/moe/utils.py (+33/-2); python/sglang/srt/models/deepseek_v2.py (+8/-4); (+10 more)
LABELS: deepseek, run-ci, jit-kernel, release-highlight
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: > Reland of #29525, reverted by #35568 because its e2e test used an ⏎ > unregistered CI runner name. This reland restores the backend and fixes that ⏎ > registration without changing unrelated CI configuration. ⏎  ⏎ ## Motivation ⏎  ⏎ Add DeepEP v2 `ElasticBuffer` as a standalone MoE A2A backend named ⏎ `deepep_v2`, alongside the existing `deepep` backend. Its fixed-capacity ⏎ communication shapes make decode CUDA-graph capturable with both single-node ⏎  …[truncated]

### L1-c967cd19b5  (L1, 2026-08-27, sha c967cd19b56b, PR #36330)
TITLE: [AMD] Optimize Qwen3.5 MTP unified attention on gfx950 (#36330)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/__init__.py (+1/-0); python/sglang/kernels/ops/attention/unified_attention_3d_mtp.py (+743/-0); python/sglang/srt/layers/attention/aiter_backend.py (+49/-0); test/registered/amd/test_unified_attention_3d_mtp.py (+113/-0)
LABELS: amd, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ Qwen3.5 target verification on MI355X uses a short multi-token query (`max_seqlen_q <= 4`) with 16 query heads, one KV head, head size 256, page size 16, and FP8 KV cache. AITER's generic 3D unified-attention configuration processes one query token per workgroup and reloads the same paged KV data for each speculative token, leaving this latency-sensitive path substantially slower than the comparison implementation. ⏎  ⏎ ## Modifica …[truncated]

### L1-b8a6adadfe  (L1, 2026-08-27, sha b8a6adadfe8c, PR #35275)
TITLE: [Bug][Spec] fix startup crash and reduce CUDA graph memory usage for speculative adaptive (#35275)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_executor/model_runner.py (+24/-3); python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py (+11/-2); python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py (+6/-1); python/sglang/srt/model_executor/runner_utils/__init__.py (+1/-0); python/sglang/srt/model_executor/runner_utils/pool.py (+13/-1); python/sglang/test/kits/attention_unittest/runner_modes/speculative_draft_runner.py (+4/-3); test/registered/unit/model_executor/test_model_runner_decode_rows.py (+51/-0)
LABELS: speculative-decoding, run-ci
BODY: ## Motivation ⏎  ⏎ Three separate problems show up when `--speculative-adaptive` is enabled: ⏎ 1. **Startup crash.** `ModelRunner.max_decode_logits_rows()` sizes the shared logits buffer from the *static* `decode_num_tokens_per_req()`, but adaptive speculative decoding replays runners built for larger draft-token widths. The buffer is too small and the server fails to start. (mentioned in #30549) ⏎ 2. **IMA on the first replay.** `DecodeCudaGraphRunn …[truncated]

### L1-3402265989  (L1, 2026-08-27, sha 3402265989c6, PR #36586)
TITLE: [Core] Refactor server argument choices (#36586)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/dsa_backend.py (+1/-1); python/sglang/srt/server_args.py (+128/-162); test/manual/test_dsa_alias_cli_registry_env.py (+18/-17); test/registered/unit/disaggregation/test_kimi_k3_encoder_mode.py (+2/-2); test/registered/unit/server_args/test_server_args.py (+42/-35)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Simplify the choice registries in `server_args.py` and keep dynamic CLI-only choices scoped to parser construction. ⏎  ⏎ ## Modifications ⏎  ⏎ - Replace trivial `add_*_choices` wrappers with bound `list.extend`/`list.append` aliases placed next to their registries. ⏎ - Inline choice lists that have no external extension surface. ⏎ - Build the token-oracle sampling choice in `ServerArgs.add_cli_args` so the environment gate is evaluated for eac …[truncated]

### L1-2ded8a6aea  (L1, 2026-08-27, sha 2ded8a6aeaeb, PR #35611)
TITLE: [AMD] Enable moe_a2a_backend=mori for DeepSeek-V4 prefill context parallelism (#35611)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/arg_groups/deepseek_v4_hook.py (+3/-3); python/sglang/srt/models/deepseek_v4.py (+7/-4)
LABELS: amd, deepseek, run-ci
BODY: Enables DeepSeek-V4 prefill context parallelism to use Mori for MoE all-to-all. ⏎  ⏎ ## Motivation ⏎  ⏎ DeepSeek-V4 prefill context parallelism rejected `moe_a2a_backend=mori`, although Mori already uses the same rank-local EP dispatch/combine path as DeepEP. ⏎  ⏎ ## Changes ⏎  ⏎ - Add `mori` to the DeepSeek-V4 CP validation whitelist. ⏎ - Add `is_mori()` to the matching model-side backend gate. ⏎  ⏎ No dispatcher, collective, or compute kernel is changed. ⏎  ⏎ ## Why thi …[truncated]

### L1-56fdfc3b26  (L1, 2026-08-27, sha 56fdfc3b2601, PR #34492)
TITLE: XPU: remove SGLANG_USE_SGL_XPU flag (#34492)
SOURCES: path_core, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+2/-4); python/sglang/srt/layers/moe/utils.py (+4/-0); python/sglang/srt/layers/attention/vision.py (+1/-2); python/sglang/srt/layers/quantization/fp8.py (+2/-5); python/sglang/srt/layers/quantization/unquant.py (+14/-11); python/sglang/srt/server_args.py (+1/-0); python/sglang/srt/utils/common.py (+0/-4); test/registered/xpu/llm_models/test_xpu_gemma_4_26b_a4b.py (+0/-1); test/registered/xpu/llm_models/test_xpu_nemotron_3_nano_30b_a3b.py (+0/-1); test/registered/xpu/llm_models/test_xpu_qwen3_30b_a3b.py (+0/-2); (+9 more)
LABELS: quant, Multi-modal, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ Remove the `SGLANG_USE_SGL_XPU` flag so that the high-performance kernels (MoE) from sgl-kernel are used by default. ⏎ If one wants to fall back to the triton kernel for MoE, use the server argument `--moe-runner-backend triton` to specify. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR …[truncated]

### L1-ad911a5ec0  (L1, 2026-08-27, sha ad911a5ec07b, PR #36543)
TITLE: [kernel] Tune LingBot-Video MoE TMA configs for H100 (#36543)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=128,N=768,device_name=NVIDIA_H100_80GB_HBM3.json (+147/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=128,N=768,device_name=NVIDIA_H100_80GB_HBM3_down.json (+147/-0); test/registered/unit/layers/moe/test_fused_moe_triton_config.py (+19/-0)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Summary ⏎  ⏎ - add Triton 3.7.1 H100 BF16 MoE configs for `E=128, N=768` ⏎ - enable TMA only for the measured 4096-token bin, for both up and down projections ⏎ - retain the current Triton 3.2 fallback values for every other token bin ⏎ - add config coverage for both projection directions and the TMA boundary ⏎  ⏎ ## H100 evidence ⏎  ⏎ Hardware/runtime: NVIDIA H100 80GB HBM3, CUDA 13.0.3, PyTorch 2.13.0+cu130, Triton 3.7.1. ⏎  ⏎ Real native SGLang workload: `robbya …[truncated]

### L1-9d07b9e227  (L1, 2026-08-27, sha 9d07b9e2278c, PR #33871)
TITLE: [Performance] Reduce idle DP work in breakable prefill CUDA graphs (#33871)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/batch_overlap/two_batch_overlap.py (+48/-7); python/sglang/srt/layers/radix_attention.py (+27/-0); python/sglang/srt/model_executor/forward_batch_info.py (+25/-6); python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py (+6/-0); python/sglang/srt/models/deepseek_v4.py (+4/-0); test/registered/unit/batch_overlap/test_tbo_children_dummy_token_mask.py (+148/-0); test/registered/unit/layers/test_radix_attention.py (+111/-0)
LABELS: deepseek, run-ci, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study performance PR ()
BODY: ## Motivation ⏎  ⏎ Under DP attention with breakable prefill CUDA graphs, idle DP ranks may be ⏎ rewritten into fabricated `EXTEND` batches so that all ranks can follow the ⏎ same execution sequence. ⏎  ⏎ Previously, the fabricated rows were counted as real tokens. As a result: ⏎  ⏎ - MoE top-k and dispatch treated dummy rows as valid tokens. ⏎ - Busy ranks performed unnecessary expert work for tokens from idle ranks. ⏎ - Attention was executed even when a …[truncated]

### L1-5adc2880f9  (L1, 2026-08-27, sha 5adc2880f996, PR #35222)
TITLE: [CPU] Enable ERNIE models on CPU (#35222)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+2/-0); python/sglang/srt/models/ernie4.py (+4/-3)
LABELS: sgl-kernel, intel, cpu, run-ci
BODY: ## Motivation ⏎  ⏎ As title. ⏎  ⏎ ## Modifications ⏎  ⏎ `topk_softmax_cpu` was implemented in #31956 which was the only lacking kernel for ERNIE. The updates in this PR are required as: ⏎  ⏎ - The `correction_bias` param is BF16 rather than FP32, so the explicit dtype conversion in `fused_topk_cpu` python intf method is needed. ⏎ - It is a 2D tensor [1, num_exp], not 1D tensor [num_exp] expected in the current kernel implementation. This squeezing is real …[truncated]

### L1-a0a2295271  (L1, 2026-08-27, sha a0a22952717a, PR #36676)
TITLE: Refactor server_args constants and layout (#36676)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/constants.py (+7/-0); python/sglang/srt/managers/scheduler_components/logprob_result_processor.py (+1/-1); python/sglang/srt/managers/tokenizer_manager_score_mixin.py (+1/-1); python/sglang/srt/server_args.py (+70/-30); test/registered/unit/managers/test_embed_overrides.py (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Keep `server_args.py` focused on server configuration and make its top-level organization easier to maintain. ⏎  ⏎ ## Modifications ⏎  ⏎ - inline single-use sampling, DeepEP v2, uvicorn, and LoRA constants ⏎ - move the multi-item-scoring delimiter sentinel to `sglang.srt.constants` ⏎ - preserve custom sampler CLI registration through a lightweight sampler registry ⏎ - document the intended top-level file structure and add section banners ⏎  ⏎ ## Acc …[truncated]

### L1-ca1d7ed8e6  (L1, 2026-08-27, sha ca1d7ed8e64c, PR #36620)
TITLE: config: a parallel leaf with no live counterpart is read bare (#36620)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.flashinfer_cutedsl, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-2); python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py (+1/-1); python/sglang/srt/layers/moe/token_dispatcher/nixl.py (+2/-2); python/sglang/srt/layers/moe/token_dispatcher/pplx.py (+1/-1); python/sglang/srt/layers/moe/utils.py (+1/-1); python/sglang/benchmark/one_batch.py (+1/-1); python/sglang/compile_deep_gemm.py (+1/-1); python/sglang/srt/batch_overlap/two_batch_overlap.py (+1/-1); python/sglang/srt/disaggregation/common/conn.py (+12/-16); python/sglang/srt/disaggregation/encoder/http_server.py (+1/-1); (+115 more)
LABELS: amd, lora, deepseek, hicache
BODY: ## Motivation ⏎  ⏎ PR 3 of a five-PR series on top of the raw-input `ServerArgs` work (#36250–#36255), based on `f775db03aaa`. Each builds on the previous one; review them in order. ⏎  ⏎ 1. `cheng/gc-p1` — config: resolution declares, and nothing writes a field ⏎ 2. `cheng/gc-p2` — config: every handler declares its cuda-graph decisions ⏎ 3. `cheng/gc-p3` — config: a parallel leaf with no live counterpart is read bare  ← **this PR** ⏎ 4. `cheng/gc-p4` — config …[truncated]

### L1-dc10483592  (L1, 2026-08-27, sha dc1048359243, PR #35374)
TITLE: [Kernel] Add H200 MoE configs for Qwen3.5 and Qwen3.6 (#35374)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=256,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+26/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=256,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json (+83/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=256,N=512,device_name=NVIDIA_H200.json (+26/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=256,N=512,device_name=NVIDIA_H200_down.json (+83/-0)
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Summary ⏎  ⏎ Add NVIDIA H200 Triton 3.6 fused-MoE configs for: ⏎  ⏎ - Qwen3.5-35B-A3B-FP8: `E=256,N=512`, block FP8 `[128,128]`, TP1, with separate up/gate and down-projection tables ⏎ - Qwen3.6-35B-A3B-FP8: the cached checkpoint requests the same lookup keys and shares the Qwen3.5 FP8 tables ⏎ - Qwen3.6-35B-A3B-BF16: `E=256,N=512`, TP1, with separate up/gate and decode-aware down-projection tables ⏎  ⏎ The unsupported Qwen3.8 MoE entry has been removed. Qwen …[truncated]

### L1-5640e53cab  (L1, 2026-08-27, sha 5640e53cab65, PR #36640)
TITLE: [NPU] [bugfix] Fix import of ggml_moe_a8_vec and Fix NPU MLA HiCache backup accessing missing data_ptrs (#36640)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py (+12/-10); python/sglang/srt/mem_cache/pool_host/mla.py (+8/-3)
LABELS: hicache, run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 36747 (confirmed_revert, reason=other)
BODY: ## Motivation ⏎  ⏎ PR #30393 generalized the MLA HiCache backup path to support packed target and draft KV buffers. However,  `NPUMLATokenToKVPool` intentionally does not create `data_ptrs` (it uses contiguous multi-layer K/V/index-K tensors for the NPU transfer kernel).  ⏎ And, Ascend NPU does not support `ggml_moe_a8_vec` which in `sgl_kernel`. ⏎  ⏎ Errors as follows： ⏎ <img width="1103" height="85" alt="error-1" src="https://github.com/user-attachme …[truncated]

### L1-76217f6603  (L1, 2026-08-27, sha 76217f660365, PR #36747)
TITLE: Revert "[NPU] [bugfix] Fix import of ggml_moe_a8_vec and Fix NPU MLA HiCache backup accessing missing data_ptrs" (#36747)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py (+10/-12); python/sglang/srt/mem_cache/pool_host/mla.py (+3/-8)
LABELS: hicache
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 36640 reason=other
BODY: Reverts sgl-project/sglang#36640 ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #33125680932](https://github.com/sgl-project/sglang/actions/runs/33125680932) ⏎ Latest PR Test (Extra): :no_entry_sign: [Run #33125680522](https://github.com/sgl-project/sglang/actions/runs/33125680522) ⏎ Latest PR Test (AMD ROCm 7.2): :no_entry_sign: [Run #33125680741](https://github.com/sgl-project/sglang/actions/runs/33125680741)

### L1-aa0a0aa3c3  (L1, 2026-08-27, sha aa0a0aa3c303, PR #36130)
TITLE: [AMD][DSV4] perf: bound the MoRI receive buffer during decode (#36130)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.runner.aiter, L1.upstream.aiter_moe
FILES: python/sglang/srt/layers/moe/moe_runner/aiter.py (+100/-0); python/sglang/srt/utils/common.py (+13/-0)
LABELS: amd, run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ AITER sizes its quantization grid from the input row count ⏎ (`aiter/csrc/kernels/quant_kernels.cu`, `rows = input.numel() / cols`). Under ⏎ MoRI that input is the **padded** receive buffer ⏎ (`num_max_dispatch_tokens_per_rank * world_size` = 4096 * 8 = 32768 rows), not ⏎ the live tokens. `num_rows` is a device pointer used only for a per-block early ⏎ exit, so it cannot shrink the grid. ⏎  ⏎ In decode the padding dominates: at concurrency 64 a  …[truncated]

### L1-2b209711d8  (L1, 2026-08-27, sha 2b209711d8d4, PR #36119)
TITLE: [AMD][DSV4] perf: MXFP8 MoRI dispatch to match the w4a8 MoE input format (#36119)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.runner.aiter, L1.ep.other_dispatchers, L1.upstream.aiter_moe
FILES: python/sglang/srt/layers/moe/moe_runner/aiter.py (+15/-1); python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+91/-2); test/registered/unit/layers/test_moriep_mxfp8_dispatch.py (+81/-0)
LABELS: amd, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ DSv4's MoE runs `per_1x32` (MXFP4 weights), so AITER wants fp8 activations ⏎ carrying group-32 e8m0 microscales. None of MoRI's three shipped dispatch dtypes ⏎ produce that: ⏎  ⏎ | dispatch | payload | scales | consequence | ⏎ | -------- | ------- | ------ | ----------- | ⏎ | bf16 | bf16 | none | receiver must quantize | ⏎ | fp8 | fp8 | group-128 fp32 | wrong group size -> fp8->bf16 upscale round trip | ⏎ | fp4 | fp4x2 | group-32 e8m0 | right scal …[truncated]

### L1-2960d69622  (L1, 2026-08-28, sha 2960d696228e, PR #36808)
TITLE: [Cookbook] Hy4-Preview follow-ups: runtime-accurate recipes + released-model info (#36808)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx (+18/-16); docs/src/snippets/configs/tencent/hy4-preview.jsx (+24/-43)
LABELS: documentation
BODY: ## Motivation ⏎  ⏎ Follow-up to #36804 (merged). Two classes of change: recipe fixes verified against the Hy4 support branch's actual runtime resolution, and content updates now that the model is publicly released. ⏎  ⏎ ## Modifications ⏎  ⏎ **Runtime-accuracy fixes** ⏎  ⏎ - **B200 MXFP8 pins `--moe-runner-backend deep_gemm`** (uniform with B300/GB300). The runtime's HYV4 family overrides default both the MoE runner and FP8 GEMM to `deep_gemm` for MXFP8 — the va …[truncated]

### L1-d3b972cbf0  (L1, 2026-08-28, sha d3b972cbf0e3, PR #36379)
TITLE: fix(lora): build the MoE LoRA align JIT kernel on ROCm (#36379)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/jit/csrc/lora/moe_lora_align_kernel.cu (+18/-1); test/registered/kernels/ops/moe/test_moe_lora_align_block_size.py (+2/-1)
LABELS: amd, lora, run-ci, jit-kernel
BODY: ## Problem ⏎  ⏎ `moe_lora_align_block_size` is JIT-compiled, and the JIT build does not run hipify. On ROCm the kernel therefore fails to compile: `cub/cub.cuh` does not exist, `cudaDevAttrMaxSharedMemoryPerBlockOptin` / `cudaFuncSetAttribute` / `cudaFuncAttributeMaxDynamicSharedMemorySize` have no HIP spellings in the JIT tree, and `hipFuncSetAttribute` takes a `const void *` rather than a function pointer. ⏎  ⏎ MoE LoRA inference reaches this kernel fr …[truncated]

### L1-69a49fede8  (L1, 2026-08-28, sha 69a49fede863, PR #36603)
TITLE: fix(kimi-k3): preserve dense ModelSlim MLA weights (#36603)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/expert_pack.py (+1/-0); python/sglang/srt/models/kimi_k3.py (+10/-3); test/registered/expert_pack/test_kimi_k3_gguf.py (+11/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ The Kimi-K3 ModelSlim W4A8 multi-node accuracy job fails after loading its 351 checkpoint shards with: ⏎  ⏎ ```text ⏎ ValueError: Kimi-K3 MLA K projection must remain GGUF Q4_0 ⏎ ``` ⏎  ⏎ Failed job: https://github.com/sgl-project/sglang/actions/runs/32936865342/job/98168349010 ⏎  ⏎ #35314 added split-GGUF K/V support for the Kimi-K3 ExpertPack path, but selected that representation through `supports_kimi_k3_quantized_latent_projections` …[truncated]

### L1-74df026877  (L1, 2026-08-28, sha 74df026877a4, PR #35677)
TITLE: fix(cpu): skip GPU JIT MoE top-k on CPU (#35677)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+6/-1)
LABELS: intel, cpu, run-ci
ISSUES: #35779 [Bug] MiniMax-M2 CPU inference fails when processing requests
DEEP_STUDY: deep-study correctness case sglang:74df026877: class=hardware_compiler_specific; symptom=crash_or_exception; introducing=#34926
BODY: ## Motivation ⏎  ⏎  ⏎ Fixes https://github.com/sgl-project/sglang/issues/35779. ⏎ [PR #34926](https://github.com/sgl-project/sglang/pull/34926/) removes the `use_jit_fused_gate` flag, forcing sigmoid MoE routing into the `biased_topk_jit_kernel_impl` (The JIT route depends on GPU-only topk_sigmoid/topk_softmax imports). ⏎ CPU device will fail during request processing, not server startup. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ * Prevent CPU from entering the GPU JI …[truncated]

### L1-c2928e86d7  (L1, 2026-08-28, sha c2928e86d78e, PR #36789)
TITLE: config: the resolution pipeline moves out of the record (#36789)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/arg_groups/attention_hook.py (+627/-0); python/sglang/srt/arg_groups/cuda_graph_hook.py (+455/-0); python/sglang/srt/arg_groups/dllm_hook.py (+124/-0); python/sglang/srt/arg_groups/hicache_hook.py (+209/-0); python/sglang/srt/arg_groups/kv_cache_hook.py (+425/-0); python/sglang/srt/arg_groups/lora_hook.py (+220/-0); python/sglang/srt/arg_groups/mamba_hook.py (+154/-0); python/sglang/srt/arg_groups/memory_hook.py (+268/-0); python/sglang/srt/arg_groups/model_hook.py (+856/-0); python/sglang/srt/arg_groups/model_path_hook.py (+306/-0); (+20 more)
LABELS: lora, Multi-modal, hicache, npu
BODY: ## Motivation ⏎  ⏎ `ServerArgs` is 11327 lines. About 3000 of those are the 483 field declarations that *are* ⏎ the record; most of the rest is the resolution pipeline living inside the same class. ⏎  ⏎ The series so far made that separable. The record holds the operator's raw input and ⏎ resolution declares rather than writes, so a decision no longer has to live next to the field ⏎ it decides — it can live next to its family. `arg_groups/` already holds seven …[truncated]

### L1-3254f9b47c  (L1, 2026-08-28, sha 3254f9b47c9f, PR #36657)
TITLE: [Blackwell] Reserve SMs for DeepGEMM MegaMoE grid barriers (#36657)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.deepgemm_megamoe
FILES: python/sglang/srt/layers/moe/mega_moe.py (+50/-11); python/sglang/srt/environ.py (+3/-0); python/sglang/srt/models/kimi_k3.py (+14/-10)
LABELS: run-ci, bypass-fastfail, run-ci-extra
ISSUES: #30399 [Bug] PD disaggregation: GB200 Deepseek v4 Pro DeepGEMM grid sync timeout
BODY: ## Summary ⏎  ⏎ - Reserve two SMs by default for Blackwell DeepGEMM MegaMoE launches. ⏎ - Scope the process-wide SM override to the actual MegaMoE kernel calls. ⏎ - Keep symmetric-buffer caching independent of the launch SM count. ⏎ - Keep Hopper/SM90 behavior unchanged and allow the reserve to be tuned or disabled. ⏎  ⏎ ## Root cause ⏎  ⏎ Blackwell MegaMoE uses an even clustered grid with a whole-grid software barrier. If the launch uses every available SM while  …[truncated]

### L1-2a96ebf648  (L1, 2026-08-28, sha 2a96ebf6486d, PR #36094)
TITLE: [AMD][DSV4] perf: retune decode split-K heuristic for MI355X (#36094)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/paged_decode.py (+28/-7); test/registered/unit/layers/test_dsv4_kv_splits_heuristic.py (+130/-0)
LABELS: run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ `_kv_splits_heuristic` picks how many KV splits DSv4 decode attention launches. ⏎ It over-split by exactly one power of two across the whole decode range: at ⏎ `H=128`/`block_h=64` it chose 8/4/2 splits for `T=32/64/128` where 4/2/1 measure ⏎ faster. ⏎  ⏎ Split-K only pays while the base grid underfills the device. Each extra split ⏎ adds a partial-buffer write plus reduce-kernel work, and once per-split K gets ⏎ short that overhead is no longer …[truncated]

### L1-4d78d59e51  (L1, 2026-08-28, sha 4d78d59e516f, PR #29718)
TITLE: [MoE] Make simulated expert routing support DP>1, and fuse into one triton kernel (#29718)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/__init__.py (+2/-0); python/sglang/srt/layers/moe/topk.py (+133/-42); python/sglang/srt/layers/moe/utils.py (+9/-0); test/manual/layers/moe/test_simulate_balanced_routing.py (+165/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Simulated expert routing is an existing feature to inject arbitrary expert choices, to isolate the effect of expert imbalance during perf benchmarking. However, the current implementation have two problems: ⏎  ⏎ 1) Does not support DP>1. When DP>1, MoE input is scattered across DP ranks, each rank generated expert assignments using the same local token indices, causing different ranks to select duplicate expert sets rather than rep …[truncated]

### L1-d12b313b93  (L1, 2026-08-28, sha d12b313b93e1, PR #36921)
TITLE: fix: KT's last MoE layer stops deferring experts again (#36921)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/kt_ep_wrapper.py (+3/-8)
DEEP_STUDY: deep-study correctness case sglang:d12b313b93: class=integration_backend_cudagraph; symptom=wrong_output_or_accuracy; introducing=#15298
BODY: ## Motivation ⏎  ⏎ `create_kt_config_from_server_args` resolves `num_layers` for one reason: so the ⏎ KT wrapper can recognise the model's final layer, which must defer no experts. ⏎  ⏎ ```python ⏎ if (self.kt_config.max_deferred_experts_per_token is not None ⏎         and self.kt_config.num_layers is not None ⏎         and self.kt_config.layer_idx == self.kt_config.num_layers - 1): ⏎     layer_max_deferred = 0 ⏎ ``` ⏎  ⏎ That branch has not fired since December 2025. ⏎  ⏎  …[truncated]

### L1-a16872767f  (L1, 2026-08-28, sha a16872767fbf, PR #36929)
TITLE: Update CUDA 13.4 image to flashinfer 0.6.18rc10, cutedsl 4.8. Fix sgl- wheel unpinning (#36929)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.cu134 (+27/-13)
BODY: ## Motivation ⏎  ⏎ * Fix bug where locally built sgl-* wheels were replaced by pip installed ones ⏎ * Upgrade to flashinfer 0.6.18 which has the jit cache for CUDA 13.4, and cutedsl 4.8 which supports sm_107 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/ …[truncated]

### L1-5f216fc33f  (L1, 2026-08-28, sha 5f216fc33f31, PR #35758)
TITLE: qwen 3.8 rebase (#35758)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.deep_gemm, L1.runner.flashinfer_trtllm, L1.runner.flashinfer_cutedsl, L1.ep.layer, L1.ep.other_dispatchers
FILES: python/sglang/kernels/ops/moe/ep_moe_kernels.py (+40/-21); python/sglang/kernels/jit/csrc/minimax/per_token_quant_ue8m0.cuh (+1/-2); python/sglang/kernels/ops/attention/triton_gdn_fused_proj.py (+306/-0); python/sglang/kernels/ops/communication/mnnvl_cutedsl/__init__.py (+52/-0); python/sglang/kernels/ops/communication/mnnvl_cutedsl/config.py (+231/-0); python/sglang/kernels/ops/communication/mnnvl_cutedsl/cute_dsl_primitives.py (+876/-0); python/sglang/kernels/ops/communication/mnnvl_cutedsl/kernel_bt/__init__.py (+43/-0); python/sglang/kernels/ops/communication/mnnvl_cutedsl/kernel_bt/device_kernels.py (+1285/-0); python/sglang/kernels/ops/communication/mnnvl_cutedsl/kernel_bt/protocol.py (+511/-0); python/sglang/kernels/ops/communication/mnnvl_cutedsl/kernel_ht/__init__.py (+37/-0); (+87 more)
LABELS: high priority, quant, dependencies, deepseek, run-ci, jit-kernel, bypass-fastfail, run-ci-extra, release-highlight
BODY: initial pr: #34585 ⏎  ⏎ ## Following work ⏎ - upgrade flashinfer to 0.6.18 ⏎ - remove all the flashinfer patches in this pr ⏎  ⏎ Note that comparing with the day 0 image, this pr is not using flashinfer gdn prefill cp kernel. ⏎  ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #33213936462](https://github.com/sgl-project/sglang/actions/runs/33213936462) ⏎ Latest PR Test (Extra): :white_check_mark: [Run #33230635480](https://github.com/sgl- …[truncated]

### L1-b644771e07  (L1, 2026-08-28, sha b644771e07e3, PR #36862)
TITLE: [Fix] Route the Mooncake MoE A2A backend through Kimi K3's EP-A2A / SP-MoE fast path (#36862)
SOURCES: path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/kimi_k3.py (+11/-7)
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Kimi K3 selects two backend-specific optimizations at construction time: ⏎  ⏎ - `self._ep_a2a` — when the MoE A2A backend routes each row to its experts directly, the MoE region consumes the rank-local rows (an SP-MoE token shard or the DP-local batch) with **no DP gather and no TP reduce**, and every global token is dispatched exactly once. ⏎ - `self._sp_moe` — on top of `_ep_a2a`, MoE layers with `attn_tp > 1` convert the deferred …[truncated]

### L1-b65e677e48  (L1, 2026-08-29, sha b65e677e489d, PR #36972)
TITLE: config: the resolution callbacks into the record go to zero (#36972)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/kt_ep_wrapper.py (+3/-1); python/sglang/srt/layers/moe/qwen35_flashinfer_fusion.py (+2/-2); python/sglang/srt/arg_groups/attention_hook.py (+15/-6); python/sglang/srt/arg_groups/cuda_graph_hook.py (+132/-15); python/sglang/srt/arg_groups/expert_pack_hook.py (+3/-1); python/sglang/srt/arg_groups/hicache_hook.py (+3/-1); python/sglang/srt/arg_groups/hisparse_hook.py (+5/-2); python/sglang/srt/arg_groups/kv_cache_hook.py (+19/-11); python/sglang/srt/arg_groups/lora_hook.py (+15/-8); python/sglang/srt/arg_groups/memory_hook.py (+125/-15); (+40 more)
LABELS: lora, Multi-modal, speculative-decoding, hicache, npu
BODY: ## The series ⏎  ⏎ Stacked, each PR based on the one above it. Read them in order; every boundary is ⏎ self-sufficient and green on its own. ⏎  ⏎ 1. [#36896](https://github.com/sgl-project/sglang/pull/36896) — the resolution pipeline's dispatcher leaves the record ⏎ 2. [#36972](https://github.com/sgl-project/sglang/pull/36972) — the callbacks into the record go to zero ← **you are here** ⏎ 3. [#36973](https://github.com/sgl-project/sglang/pull/36973) — six mor …[truncated]

### L1-1a3e152f03  (L1, 2026-08-29, sha 1a3e152f03b5, PR #36973)
TITLE: config: six more runtime readers ask the bags (#36973)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/kt_ep_wrapper.py (+8/-7); python/sglang/srt/elastic_ep/expert_backup_client.py (+5/-8); python/sglang/srt/eplb/expert_distribution.py (+12/-29); python/sglang/srt/managers/prefill_delayer.py (+3/-4); python/sglang/srt/managers/scheduler.py (+3/-4); python/sglang/srt/managers/scheduler_components/metrics_reporter.py (+4/-7); python/sglang/srt/managers/tokenizer_manager.py (+0/-1); python/sglang/srt/mem_cache/base_prefix_cache.py (+0/-4); python/sglang/srt/mem_cache/hiradix_cache.py (+0/-2); python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py (+6/-6); (+10 more)
LABELS: documentation, quant, amd, deepseek, speculative-decoding, hicache, sgl-kernel, model-gateway, unified-radix-cache
BODY: ## The series ⏎  ⏎ Stacked, each PR based on the one above it. Read them in order; every boundary is ⏎ self-sufficient and green on its own. ⏎  ⏎ 1. [#36896](https://github.com/sgl-project/sglang/pull/36896) — the resolution pipeline's dispatcher leaves the record ⏎ 2. [#36972](https://github.com/sgl-project/sglang/pull/36972) — the callbacks into the record go to zero ⏎ 3. [#36973](https://github.com/sgl-project/sglang/pull/36973) — six more runtime readers a …[truncated]

### L1-4d53767b09  (L1, 2026-08-29, sha 4d53767b0942, PR #36975)
TITLE: config: the lazy imports that buy nothing become eager (#36975)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.marlin
FILES: python/sglang/srt/layers/moe/kt_ep_wrapper.py (+1/-2); python/sglang/srt/arg_groups/attention_hook.py (+17/-30); python/sglang/srt/arg_groups/cuda_graph_hook.py (+3/-6); python/sglang/srt/arg_groups/deepseek_v4_hook.py (+2/-4); python/sglang/srt/arg_groups/dllm_hook.py (+4/-7); python/sglang/srt/arg_groups/expert_pack_hook.py (+5/-2); python/sglang/srt/arg_groups/hicache_hook.py (+1/-1); python/sglang/srt/arg_groups/hisparse_hook.py (+5/-6); python/sglang/srt/arg_groups/kv_cache_hook.py (+3/-2); python/sglang/srt/arg_groups/memory_hook.py (+3/-6); (+28 more)
LABELS: documentation, quant, lora, Multi-modal, deepseek, speculative-decoding, hicache, npu, diffusion, jit-kernel
BODY: ## The series ⏎  ⏎ Stacked, each PR based on the one above it. Read them in order; every boundary is ⏎ self-sufficient and green on its own. ⏎  ⏎ 1. [#36896](https://github.com/sgl-project/sglang/pull/36896) — the resolution pipeline's dispatcher leaves the record ⏎ 2. [#36972](https://github.com/sgl-project/sglang/pull/36972) — the callbacks into the record go to zero ⏎ 3. [#36973](https://github.com/sgl-project/sglang/pull/36973) — six more runtime readers a …[truncated]

### L1-00fbb6e8ac  (L1, 2026-08-29, sha 00fbb6e8ace3, PR #35760)
TITLE: [Perf] Tune the W4AFP8 DeepEP low-latency requant launch geometry (#35760)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.ep.layer, L1.cutlass.adapters
FILES: python/sglang/kernels/ops/moe/ep_moe_kernels.py (+206/-41); python/sglang/srt/layers/moe/cutlass_w4a8_moe.py (+4/-0); python/sglang/srt/layers/quantization/w4afp8.py (+4/-1); test/registered/kernels/benchmark/moe/bench_fp8_per_token_to_per_tensor_quant.py (+135/-0); test/registered/kernels/ops/moe/test_fp8_per_token_to_per_tensor_quant.py (+124/-33); test/registered/unit/layers/moe/test_w4afp8_requant_geometry.py (+145/-0)
LABELS: quant, run-ci, jit-kernel, run-ci-extra
DEEP_STUDY: deep-study performance PR ()
BODY: ## Motivation ⏎  ⏎ `fp8_per_token_to_per_tensor_quant_triton()` turns DeepEP low-latency's per-token-group fp8 ⏎ payload into the per-tensor fp8 that the first CUTLASS W4A8 grouped GEMM consumes. It is a pure ⏎ streaming pass — two bytes moved and two multiplies per element — but three properties of its ⏎ launch shape cost more than the arithmetic does. ⏎  ⏎ **Access width.** A 1024-element tile over 8 warps is 4 B/thread, which is exactly where Triton ⏎ stops v …[truncated]

### L1-fbecd75c83  (L1, 2026-08-29, sha fbecd75c8313, PR #36515)
TITLE: [AMD] fix: do not emit a shared-expert marker twice on the per-rank slot path (#36515)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+14/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ On the per-rank fused shared-slot path the shared expert is appended after the ⏎ gate, by `fused_append_remap_shared_experts_deepep`. But `select_experts` also ⏎ passes `num_fused_shared_experts` down into the gate, so both of them emit a ⏎ shared marker. ⏎  ⏎ Two things go wrong, neither of which raises: ⏎  ⏎ **A routed expert is lost.** The gate is asked for ⏎ `K_routed = top_k - num_fused_shared_experts` slots and then spends one of them ⏎ on its …[truncated]

### L1-ed39568e79  (L1, 2026-08-29, sha ed39568e794f, PR #32665)
TITLE: [MoE] Add extension points for custom runner backends (#32665)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.framework
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+10/-7); python/sglang/srt/layers/moe/moe_runner/__init__.py (+2/-2); python/sglang/srt/layers/moe/moe_runner/base.py (+21/-0); python/sglang/srt/layers/moe/moe_runner/runner.py (+52/-16); python/sglang/srt/layers/moe/utils.py (+91/-45); python/sglang/srt/layers/quantization/base_config.py (+13/-1); python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py (+4/-0); python/sglang/srt/lora/lora_moe_runner_marlin.py (+7/-2); python/sglang/srt/lora/layers.py (+14/-14); test/registered/unit/layers/moe/test_moe_runner_extensions.py (+378/-0); (+2 more)
LABELS: lora, run-ci, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ Allow out-of-tree MoE backends to use SGLang’s standard routing and `MoeRunner` abstraction without adding private backend names to the public enum. ⏎  ⏎ ## Modifications ⏎  ⏎ - Add registered backend names and dispatch-native runner-core factories. ⏎ - Allow `FusedMoE` to receive an explicit quant method. ⏎ - Add a backend-neutral MoE quant-info contract used by LoRA runners. ⏎ - Route Marlin LoRA through the standard dispatch interfac …[truncated]

### L1-a7ee399904  (L1, 2026-08-29, sha a7ee39990456, PR #36998)
TITLE: Bump sgl-deep-gemm to v0.1.6 (#36998)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1)
LABELS: dependencies, run-ci, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Rebased on latest DeepGemm official main branch ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/b …[truncated]

### L1-09ecb9aaaa  (L1, 2026-08-29, sha 09ecb9aaaab9, PR #36768)
TITLE: :memo: [NPU] Use vendor-neutral wording in quantization comments (#36768)
SOURCES: path_core
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/hardware_backend/npu/moe/init_routing.py (+1/-2); python/sglang/multimodal_gen/runtime/layers/quantization/modelslim_mxfp4_scheme.py (+1/-7); python/sglang/multimodal_gen/runtime/layers/quantization/modelslim_mxfp8_scheme.py (+1/-1); python/sglang/multimodal_gen/runtime/layers/quantization/mxfp4_npu.py (+10/-13); python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py (+11/-12); python/sglang/srt/hardware_backend/npu/quantization/moe_methods.py (+5/-5); python/sglang/srt/hardware_backend/npu/quantization/online_moe_methods.py (+2/-2); python/sglang/srt/models/qwen3_moe.py (+3/-3)
LABELS: npu, diffusion
BODY: ## Motivation ⏎  ⏎ Follow-up to #34829. That PR cleaned the MXFP8/MXFP4 linear methods, but the ⏎ NPU quantization comments still carried two kinds of vendor wording, all landed ⏎ by my earlier NPU quantization PRs: ⏎  ⏎ 1. Comments naming external projects, left in five files #34829 did not reach. ⏎ 2. `Ascend` used as prose branding in docstrings for modules that are otherwise ⏎    named and namespaced `npu`. ⏎  ⏎ ## Modifications ⏎  ⏎ Comment-only cleanup …[truncated]

### L1-7e751153eb  (L1, 2026-08-30, sha 7e751153eb59, PR #37086)
TITLE: [Config] Round 5.1: the published-side readers ask the bags, and a platform fact gets one address (#37086)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.runner.framework, L1.runner.deep_gemm, L1.runner.openai_triton_kernels, L1.runner.aiter, L1.ep.deepep_dispatcher, L1.ep.other_dispatchers, L1.upstream.openai_triton_kernels, L1.upstream.aiter_moe, L1.cutlass.adapters
FILES: python/sglang/srt/arg_groups/attention_hook.py (+12/-18); python/sglang/srt/arg_groups/cuda_graph_hook.py (+7/-6); python/sglang/srt/arg_groups/deepseek_v4_hook.py (+2/-2); python/sglang/srt/arg_groups/dllm_hook.py (+2/-2); python/sglang/srt/arg_groups/hisparse_hook.py (+4/-10); python/sglang/srt/arg_groups/kimi_k3_hook.py (+3/-4); python/sglang/srt/arg_groups/kv_cache_hook.py (+5/-13); python/sglang/srt/arg_groups/mamba_hook.py (+10/-14); python/sglang/srt/arg_groups/model_hook.py (+16/-17); python/sglang/srt/arg_groups/moe_hook.py (+3/-2); (+138 more)
LABELS: quant, amd, Multi-modal, deepseek, speculative-decoding, hicache, blackwell, npu, unified-radix-cache
BODY: ## Motivation ⏎  ⏎ Round 4 got `arg_groups` to the point where resolution never writes a field on ⏎ `ServerArgs`: handlers *declare* what they decide, and `publish()` projects the ⏎ result into read-only config bags. The reader side was still half-converted — ⏎ runtime code read raw `server_args` fields, two production names existed only so ⏎ tests could patch them, and a platform fact such as "this box is SM100" had one ⏎ copy per importing module, so it coul …[truncated]

### L1-e51a3ae65e  (L1, 2026-08-30, sha e51a3ae65e34, PR #37087)
TITLE: [Config] Round 5.2: the per-model declarations get their own modules (#37087)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/arg_groups/model_override_base.py (+376/-0); python/sglang/srt/arg_groups/model_overrides/__init__.py (+37/-0); python/sglang/srt/arg_groups/model_overrides/deepseek_v2.py (+145/-0); python/sglang/srt/arg_groups/model_overrides/deepseek_v4.py (+76/-0); python/sglang/srt/arg_groups/model_overrides/exaone.py (+23/-0); python/sglang/srt/arg_groups/model_overrides/falcon_h1.py (+22/-0); python/sglang/srt/arg_groups/model_overrides/gemma2_gemma3.py (+29/-0); python/sglang/srt/arg_groups/model_overrides/gemma4.py (+47/-0); python/sglang/srt/arg_groups/model_overrides/glm4_moe.py (+50/-0); python/sglang/srt/arg_groups/model_overrides/gpt_oss.py (+127/-0); (+34 more)
LABELS: deepseek, bypass-fastfail, run-ci-extra
BODY: > **Stacked on #37086 — merge that first.** Until it lands, the diff shown here ⏎ > includes part 1; only the last 7 commits belong to this PR. ⏎  ⏎ ## Motivation ⏎  ⏎ The 2026-07-03 placement ruling said the config-time model overrides stay in ⏎ `arg_groups/` — the model modules are not imported at `__post_init__`, and the ⏎ frontend process must not depend on a GPU kernel import — but that they could ⏎ split out mechanically once the families grew. 27 handlers …[truncated]

### L1-4f761e8649  (L1, 2026-08-30, sha 4f761e8649c4, PR #36954)
TITLE: [Deps] Bump FlashInfer to 0.6.18 (#36954)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); docker/kimi_k3/apply_deepep_k3_patch.sh (+0/-169); docker/kimi_k3/kimi_k3_cu12.Dockerfile (+0/-113); docker/kimi_k3/kimi_k3_cu13.Dockerfile (+0/-101); python/pyproject.toml (+1/-1); docker/Dockerfile.cu134 (+1/-1); docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/utils/common.py (+1/-1); test/registered/unit/mem_cache/test_unified_mamba_views.py (+4/-4)
LABELS: documentation, dependencies, run-ci, bypass-fastfail, run-ci-extra, release-highlight
BODY: --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :white_check_mark: [Run #33289293501](https://github.com/sgl-project/sglang/actions/runs/33289293501) ⏎ Latest PR Test (Extra): :white_check_mark: [Run #33349424388](https://github.com/sgl-project/sglang/actions/runs/33349424388) ⏎ Latest PR Test (AMD ROCm 7.2): :x: [Run #33289293415](https://github.com/sgl-project/sglang/actions/runs/33289293415)

### L1-3865efc9f7  (L1, 2026-08-31, sha 3865efc9f7e8, PR #36871)
TITLE: [AMD] support gfx1250 on ROCM 10 (#36871)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.fused_gate, L1.runner.aiter
FILES: docker/rocm.Dockerfile (+158/-43); python/sglang/kernels/jit/csrc/moe/moe_fused_gate.cuh (+15/-6); python/sglang/srt/layers/moe/fused_moe_triton/aiter_mxfp4_w4a8_moe.py (+210/-0); .github/workflows/release-docker-amd-rocm7_15-nightly.yml (+0/-109); python/pyproject_other.toml (+1/-5); python/sglang/kernels/aot/csrc/allreduce/quick_all_reduce_base.h (+6/-1); python/sglang/kernels/aot/setup_rocm.py (+1/-1); python/sglang/kernels/ops/attention/decode_attention.py (+33/-4); python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/paged_decode.py (+28/-9); python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/paged_prefill.py (+2/-2); (+25 more)
LABELS: high priority, amd, dependencies, deepseek, sgl-kernel, run-ci, jit-kernel, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ Based on https://github.com/sgl-project/sglang/pull/32754 (gfx1250 enablement) and https://github.com/sgl-project/sglang/pull/36434 (ROCm 10 release images), see those PRs for the details of each. ⏎  ⏎ On top of them, this adds the gfx1250-rocm1000 Docker target on the ROCm 10.0.0 GA wheel channel, gfx1250 detection in the AMD CI dependency installer, and the gfx1250 kernel/model fixes with the MI45x accuracy tests. ⏎  ⏎ ## Modificat …[truncated]

### L1-3139ceaeec  (L1, 2026-08-31, sha 3139ceaeec50, PR #33318)
TITLE: [XPU] Use SYCL kernels for topk_transform on XPU (#33318)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/dsv4/topk.py (+24/-0); python/sglang/srt/layers/attention/dsv4/metadata.py (+2/-2)
LABELS: deepseek, intel, xpu, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ Route `topk_transform_512` and `topk_transform_512_v2` to the native SYCL ⏎ kernels on XPU. The SYCL implementations are added in the `sgl-kernel-xpu` PR (https://github.com/sgl-project/sgl-kernel-xpu/pull/366) and exposed as ⏎ `torch.ops.sgl_kernel.topk_transform_512{,_v2}`. Without this dispatch, XPU ⏎ runs fall through to the JIT CUDA path and fail to load, or drop to a ⏎ significantly slower vectorized PyTorch fallback. ⏎  ⏎ ## Mod …[truncated]

### L1-88cf5c9541  (L1, 2026-08-31, sha 88cf5c954193, PR #37293)
TITLE: [Cookbook] Add DeepSeek-V4-Flash-Vision-Exp to the DeepSeek-V4 page (#37293)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .claude/skills/cookbook-add-model/SKILL.md (+4/-2); .claude/skills/cookbook-add-model/references/authoring-reference.md (+1/-1); .claude/skills/cookbook-add-model/templates/config.jsx.tmpl (+8/-6); docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx (+70/-5); docs/src/snippets/_deployment.jsx (+9/-4); docs/src/snippets/_playground.jsx (+10/-6); docs/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx (+24/-0); docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx (+291/-2)
LABELS: documentation, deepseek
BODY: ## Motivation ⏎  ⏎ Add the experimental multimodal checkpoint [`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp) (engine support: #37253) to the DeepSeek-V4 cookbook as a new **Flash Vision** variant on the existing page. ⏎  ⏎ ## Modifications ⏎  ⏎ **Deploy matrix (B200 · FP4)** ⏎ - `low-latency`: **verified** end-to-end — exactly the shape the MMMU-Pro round ran on 4×B200 (`--tp 4 --mem-fraction-stati …[truncated]

### L1-07c8f7294d  (L1, 2026-08-31, sha 07c8f7294daa, PR #37279)
TITLE: Bump sgl-deep-gemm to 0.1.7 (#37279)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L1.runner.deepgemm_megamoe, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/arg_groups/mega_moe_hook.py (+0/-11); python/sglang/srt/layers/moe/mega_moe.py (+28/-11); python/sglang/srt/layers/quantization/mxfp4.py (+4/-0); python/sglang/srt/server_args.py (+2/-2); test/registered/unit/layers/moe/test_mega_moe_deepgemm_api.py (+262/-0); test/registered/unit/server_args/test_server_args.py (+3/-3)
LABELS: dependencies, run-ci
BODY: ## Motivation ⏎ To include nvfp4 megamoe support ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/ …[truncated]

### L1-9a85473a89  (L1, 2026-08-31, sha 9a85473a8959, PR #35120)
TITLE: [FlashInfer v0.6.18] add FlashInfer CuTe DSL NVFP4 W4A16 mode (#35120)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.runner.flashinfer_cutedsl
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py (+30/-9); docs/docs/references/environment_variables.mdx (+5/-0); python/sglang/srt/arg_groups/moe_hook.py (+11/-1); python/sglang/srt/environ.py (+2/-0); python/sglang/srt/layers/logits_processor.py (+10/-0); python/sglang/srt/layers/quantization/modelopt_quant.py (+136/-56); python/sglang/srt/models/deepseek_v2.py (+1/-0); python/sglang/srt/models/qwen3_5.py (+1/-0); test/registered/backends/test_flashinfer_nvfp4_online_moe_backend.py (+43/-1); test/registered/rl/test_update_weights_from_disk_blackwell.py (+33/-1); (+2 more)
LABELS: documentation, quant, deepseek, blackwell, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ @humansand ⏎  ⏎ Add an opt-in W4A16 compute mode for FlashInfer CuTe DSL NVFP4 MoE and dense linear layers: weights remain NVFP4 while activations and outputs stay BF16. Registered tests cover online-quantized Nemotron MoE accuracy, serialized ModelOpt Qwen3 dense+MoE disk-reload invariance, and the existing dense W4A4 numerical path. ⏎  ⏎ Dependency: ⏎  ⏎ - This PR requires the final [FlashInfer v0.6.18](https://github.com/flashinfer-ai/flash …[truncated]

### L1-97744189b8  (L1, 2026-08-31, sha 97744189b81d, PR #37317)
TITLE: [Kernel] Raise shape limits in shared FLA and MoE kernels (ported from #36507) (#37317)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/jit/csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh (+34/-12); python/sglang/kernels/ops/attention/fla/cumsum.py (+1/-2); python/sglang/kernels/ops/attention/fla/fused_recurrent.py (+2/-3); python/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py (+21/-13); python/sglang/srt/layers/attention/vision.py (+1/-2)
LABELS: quant, Multi-modal, jit-kernel
BODY: Ported from #36507. ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :x: [Run #33459097951](https://github.com/sgl-project/sglang/actions/runs/33459097951) ⏎ Latest PR Test (Extra): :x: [Run #33459097775](https://github.com/sgl-project/sglang/actions/runs/33459097775) ⏎ Latest PR Test (AMD ROCm 7.2): :x: [Run #33459097938](https://github.com/sgl-project/sglang/actions/runs/33459097938)

### L1-5b04408784  (L1, 2026-08-31, sha 5b0440878431, PR #34967)
TITLE: [MoE] Add FlashInfer SM90 MXFP4 W4A8 CUTLASS MoE (#34967)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.flashinfer_cutlass
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+5/-0); python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py (+42/-2); python/sglang/srt/layers/quantization/mxfp4.py (+109/-29); python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py (+65/-20); python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py (+3/-0); python/sglang/srt/server_args.py (+4/-2); docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx (+1/-1); docs/docs/advanced_features/server_arguments.mdx (+2/-2); docs/docs/hardware-platforms/ascend-npus/reference/support_features.mdx (+1/-1); docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx (+13/-5); (+3 more)
LABELS: quant, deepseek, run-ci, hopper, flashinfer, run-ci-extra, release-highlight
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ SGLang currently supports FlashInfer's SM90 MXFP4 W4A16 CUTLASS MoE path, which was introduced in SGLang #24816 on top of FlashInfer [#3084](https://github.com/flashinfer-ai/flashinfer/pull/3084). FlashInfer [#3738](https://github.com/flashinfer-ai/flashinfer/pull/3738) adds a different Hopper path that dynamically quantizes activations to FP8 and runs MXFP4-weight x FP8-activation MoE GEMMs using Humming-style pre-MMA E8M0 scale …[truncated]

### L1-d60d658f5f  (L1, 2026-08-31, sha d60d658f5f0f, PR #37159)
TITLE: [Kernel] Add GB300 Triton MoE configs for GLM-4.5 FP8 (#37159)
SOURCES: path_config_only, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=161,N=192,device_name=NVIDIA_GB300,dtype=fp8_w8a8,per_channel_quant=True.json (+58/-0)
LABELS: quant, run-ci
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Motivation ⏎  ⏎ GLM-4.5-FP8 uses per-channel FP8 Triton MoE with the per-rank TP8 geometry ⏎ `E=161,H=5120,I=192,topk=9`. SGLang had no Triton 3.7.1 GB300 config for this ⏎ exact lane and therefore used the generic heuristic. ⏎  ⏎ The missing tuned baseline was identified while validating ⏎ [KDA-Pilot PR #197](https://github.com/BBuf/KDA-Pilot/pull/197). A larger custom ⏎ kernel was also evaluated, but the tuned generic kernel was faster, so this PR ⏎ only adds …[truncated]

### L1-3315356cc0  (L1, 2026-09-01, sha 3315356cc043, PR #37360)
TITLE: docs(cookbook): enable FlashInfer GDN for Qwen3.5 B200 (#37360)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/src/snippets/autoregressive/qwen35-deployment.jsx (+9/-0)
LABELS: documentation
BODY: ## Motivation ⏎  ⏎ The Qwen3.5-397B NVFP4 B200 MTP recipe runs with TP2/EP2. The exact configuration was validated by [SemiAnalysisAI/InferenceX#2790](https://github.com/SemiAnalysisAI/InferenceX/pull/2790) in [Run Sweep 33451222974, attempt 2](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/33451222974) on head `0c32f3c2ba871287587bb1baf0d21cee65c44bd4`. ⏎  ⏎ This change carries the validated Gated Delta Network backend routing into the genera …[truncated]

### L1-0f18d389b4  (L1, 2026-09-01, sha 0f18d389b46b, PR #37468)
TITLE: [Cookbook] Verify DeepSeek-V4 Flash Vision balanced and high-throughput on B200 (#37468)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx (+12/-2); docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx (+2/-4)
LABELS: documentation, deepseek, run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Follow-up to #37293 / #37301: the B200 Flash Vision **balanced** (DeepEP) and **high-throughput** (MegaMoE) cells shipped as `verificationStatus: "in-progress"` with bare benchmark stubs. This PR lands their verification round. ⏎  ⏎ ## Verification ⏎  ⏎ Both cells were run **verbatim** on 4(×2)×B200 (TP=4, DP=4) with the cookbook MMMU-Pro command (`sgl-eval run mmmu_pro --reasoning-effort max --temperature 1.0 --top-p 0.95`), on `lmsysorg/ …[truncated]

### L1-e57e934bcc  (L1, 2026-09-01, sha e57e934bcc5c, PR #32882)
TITLE: [Bugfix] Accept int64 top-k IDs in FlashInfer routed MoE packer (#32882)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/moe/pack_topk_ids.py (+6/-3)
LABELS: bug, run-ci, jit-kernel
BODY: [by Codex] ⏎  ⏎ ## Summary ⏎  ⏎ - accept both `torch.int32` and `torch.int64` router IDs in `PackTopkIds` ⏎ - keep the existing in-kernel conversion to int32, avoiding a temporary cast allocation during CUDA-graph capture ⏎ - add focused coverage for both input dtypes, multiple shapes, exact reference parity, CUDA-graph capture/replay, and invalid input ⏎  ⏎ This fixes a Qwen3 NVFP4 startup crash on Blackwell when torch compile uses the native top-k path: ⏎  ⏎ ```te …[truncated]

### L1-221a6273ce  (L1, 2026-09-01, sha 221a6273ce32, PR #36811)
TITLE: [Kernel] Avoid zero-bias allocation in fused softmax routing (#36811)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py, L1.routing.fused_gate
FILES: python/sglang/kernels/ops/moe/moe_fused_gate.py (+29/-17); python/sglang/srt/layers/moe/topk.py (+1/-6); test/registered/kernels/ops/moe/test_moe_fused_gate.py (+17/-0)
LABELS: run-ci, jit-kernel, bypass-fastfail
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Summary ⏎  ⏎ Plain softmax routing has no correction bias, but `fused_topk` currently allocates and clears a device-side zero tensor on every call to satisfy the fused router kernel interface. This PR represents the same operation with a compile-time no-bias path, avoiding the allocation, fill, and bias load while preserving the existing sigmoid/correction-bias path. ⏎  ⏎ ## Changes ⏎  ⏎ - Allow `moe_fused_gate` to receive `bias=None` for softmax routing. …[truncated]

### L1-6d34a4d3ce  (L1, 2026-09-01, sha 6d34a4d3ce10, PR #37492)
TITLE: [Cookbook] Verify DeepSeek-V4 Flash Vision on GB300 (#37492)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx (+19/-4); docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx (+5/-7)
LABELS: documentation, deepseek, run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Follow-up to #37293 / #37301 / #37468: the GB300 Flash Vision cells shipped as `verificationStatus: "in-progress"` with bare benchmark stubs. This PR lands their verification round — the first non-B200 hardware row for this variant. ⏎  ⏎ ## Verification ⏎  ⏎ All three cells were run **verbatim** on 4×GB300 (arm64) with the cookbook MMMU-Pro command (`sgl-eval run mmmu_pro --reasoning-effort max --temperature 1.0 --top-p 0.95`), on `lmsysor …[truncated]

### L1-c66a285c94  (L1, 2026-09-01, sha c66a285c94bf, PR #37477)
TITLE: [Kernel] GLM 5.3 Flash related kernels (ported from #36507) (#37477)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/moe/kpool_topk_transform.py (+71/-0); python/sglang/kernels/jit/csrc/dsa/kpool_topk_transform.cuh (+412/-0); python/sglang/kernels/ops/attention/__init__.py (+1/-0); python/sglang/kernels/ops/attention/dsa/transform_index.py (+52/-0); python/sglang/kernels/ops/attention/fla/kda.py (+5/-0); python/sglang/kernels/ops/attention/helion/kda_prefill.py (+3/-0); python/sglang/kernels/ops/attention/utils.py (+21/-0); python/sglang/kernels/ops/kvcache/mla_buffer.py (+94/-5); python/sglang/kernels/ops/layernorm/mhc.py (+184/-0); python/sglang/srt/layers/attention/dsa/kpool_fp8_index.py (+1701/-0); (+6 more)
LABELS: run-ci, jit-kernel, run-ci-extra
BODY: Ported from #36507. ⏎  ⏎ No behavior change on main: new files with no callers, plus additive parameters that default to existing behavior. ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #33571827221](https://github.com/sgl-project/sglang/actions/runs/33571827221) ⏎ Latest PR Test (Extra): :white_check_mark: [Run #33571826901](https://github.com/sgl-project/sglang/actions/runs/33571826901) ⏎ Latest PR Test (AMD ROCm 7.2): :hourglass_flow …[truncated]

### L1-b6c06e1efb  (L1, 2026-09-01, sha b6c06e1efb60, PR #36831)
TITLE: [DSA] Drop the redundant 512 from the top-k transform entry-point names (#36831)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/dsv4/topk.py (+4/-4); python/sglang/kernels/ops/attention/dsv4/__init__.py (+3/-3); python/sglang/srt/arg_groups/model_hook.py (+1/-1); python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py (+2/-2); python/sglang/srt/layers/attention/dsv4/indexer.py (+14/-14); test/registered/kernels/benchmark/attention/bench_topk.py (+4/-4); test/registered/kernels/ops/attention/test_topk_v2.py (+3/-3)
LABELS: run-ci, jit-kernel, run-ci-extra
BODY: > Generated by Claude. ⏎  ⏎ ## Motivation ⏎  ⏎ The `512` in `topk_transform_512_v2` (and its v1 / fallback siblings) is a leftover from when the entry points were specialized for `topk == 512`. Both kernels have since taken `topk` as a runtime argument -- v1 up to 1024, v2 up to 2048 -- so the number in the name is now simply wrong for every production shape (DSA runs 2048). ⏎  ⏎ ## Modifications ⏎  ⏎ Pure rename, no behavior change. The paged entry points are n …[truncated]

### L1-ee462b5899  (L1, 2026-09-01, sha ee462b5899c0, PR #37158)
TITLE: [Kernel] Add tuned LFM2.5 Triton MoE configs on B300 (#37158)
SOURCES: path_config_only
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=32,N=1792,device_name=NVIDIA_B300_SXM6_AC.json (+74/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=32,N=1792,device_name=NVIDIA_B300_SXM6_AC_down.json (+74/-0)
LABELS: run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Motivation ⏎  ⏎ LFM2.5-8B-A1B uses two BF16 Triton MoE GEMMs on B300 (`SM103`): ⏎  ⏎ - gate/up projection: `E=32, N=3584, K=2048, topk=4` ⏎ - down projection: `E=32, N=2048, K=1792, topk=1` ⏎  ⏎ The generic Triton path previously fell back to heuristic schedules for this ⏎ model/device pair. A shape-specific KDA kernel improved over that fallback, but a ⏎ separately tuned generic Triton baseline was faster for medium/large token counts ⏎ and high-concurrency serv …[truncated]

### L1-acea43079f  (L1, 2026-09-02, sha acea43079fa5, PR #36407)
TITLE: Fix native MoE handling of noncontiguous top-k IDs (#36407)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_native.py (+1/-1); test/registered/unit/layers/moe/test_fused_moe_native.py (+55/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ The native MoE path flattens `topk_ids` with `view`, which requires a stride-compatible tensor. Routing implementations may return valid noncontiguous top-k ID tensors, causing expert dispatch to fail even though the computation does not require contiguous storage. ⏎  ⏎ ## Modifications ⏎  ⏎ - Replace `topk_ids.view(-1)` with `topk_ids.reshape(-1)` in the native MoE path. ⏎ - Preserve zero-copy behavior for stride-compatible layouts while ma …[truncated]

### L1-19c30dff56  (L1, 2026-09-02, sha 19c30dff5620, PR #29927)
TITLE: [SM120] DeepSeek-V4: DeepGEMM paged-MQA indexer +FP4 MoE+ page-split  (#29927)
SOURCES: path_core, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.runner.deep_gemm
FILES: python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+29/-2); python/sglang/srt/layers/moe/moe_runner/deep_gemm_sm120.py (+198/-0); python/sglang/kernels/ops/attention/flash_mla_sm120.py (+98/-19); python/sglang/kernels/ops/layernorm/mhc.py (+28/-3); python/sglang/srt/arg_groups/model_hook.py (+12/-5); python/sglang/srt/layers/attention/deepseek_v4_backend.py (+12/-0); python/sglang/srt/layers/attention/dsv4/indexer.py (+101/-77); python/sglang/srt/layers/attention/dsv4/metadata.py (+24/-7); python/sglang/srt/layers/deep_gemm_wrapper/configurer.py (+9/-3); python/sglang/srt/model_loader/utils.py (+5/-0); (+1 more)
LABELS: quant, deepseek, run-ci, jit-kernel, run-ci-extra, release-highlight
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: **Motivation:** a 4x RTX PRO 6000 Blackwell (SM120) box can hold DeepSeek-V4-Flash, but on ⏎ current main the only configuration that starts there is the slow torch fallback — this PR ⏎ makes the DeepGEMM and FlashInfer paths work on SM120, cutting decode latency up to 3.4x. ⏎  ⏎ It routes the sparse-MLA indexer to DeepGEMM's paged-MQA-logits kernel, enables batched ⏎ sparse-MLA prefill through FlashInfer, and unblocks the DeepGEMM FP4 MoE backend. ⏎  ⏎ ## Spl …[truncated]

### L1-28262c20df  (L1, 2026-09-02, sha 28262c20df6f, PR #37210)
TITLE: [CI][RFC] Replace black-jupyter with ruff-format (#37210)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels, L1.routing.topk_py, L1.routing.fused_gate, L1.routing.hash_topk, L1.runner.framework, L1.runner.triton, L1.runner.deep_gemm, L1.runner.openai_triton_kernels, L1.runner.flashinfer_trtllm, L1.runner.flashinfer_cutedsl, L1.runner.flashinfer_cutlass, L1.runner.marlin, L1.runner.aiter, L1.runner.deepgemm_megamoe, L1.ep.layer, L1.ep.other_dispatchers, L1.upstream.openai_triton_kernels, L1.upstream.aiter_moe, L1.cutlass.adapters
FILES: .claude/skills/llm-torch-profiler-analysis/scripts/triage_kernel_helpers.py (+4/-6); .claude/skills/mechanical-refactor-verify/scripts/mechanical_refactor_proof_generator.py (+3/-1); .claude/skills/mechanical-refactor-verify/scripts/mechanical_refactor_reproduction_utils.py (+33/-29); .claude/skills/mechanical-refactor-verify/scripts/tests/proof_generator/test_infer_moves.py (+3/-12); .claude/skills/mechanical-refactor-verify/scripts/tests/reproduction_cli/cli_testlib.py (+2/-6); .claude/skills/mechanical-refactor-verify/scripts/tests/reproduction_utils/test_add_imports.py (+2/-13); .claude/skills/mechanical-refactor-verify/scripts/tests/reproduction_utils/test_extract_symbols_to_new_module.py (+1/-5); .claude/skills/mechanical-refactor-verify/scripts/tests/reproduction_utils/test_move_assign.py (+2/-14); .claude/skills/mechanical-refactor-verify/scripts/tests/reproduction_utils/test_move_symbol.py (+1/-7); .claude/skills/sglang-prod-incident-triage/scripts/incident_artifact_tool.py (+4/-3); (+1401 more)
LABELS: documentation, high priority, quant, amd, lora, Multi-modal, deepseek, speculative-decoding, hicache, sgl-kernel
BODY: ## Motivation ⏎  ⏎ black-jupyter is now the dominant cost of the lint job: **218s** of the ~6min job (measured on run [33360547828](https://github.com/sgl-project/sglang/actions/runs/33360547828/job/99390908592)), up from 143s in June — it re-formats every file each run, scales with the repo, and swings ±20% with runner CPU (observed 218s→260s across two runs of the same commit). Its on-disk cache can never work in CI (keyed on mtime; every fresh che …[truncated]

### L1-a6001478f4  (L1, 2026-09-03, sha a6001478f430, PR #33838)
TITLE: [AMD] Perf Kimi-K3 MoE optimization (#33838)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+30/-1); python/sglang/srt/layers/quantization/mxfp4.py (+26/-9); test/registered/unit/layers/moe/test_topk_correction_bias_cache.py (+72/-0); test/registered/unit/layers/quantization/test_mxfp4_situ_output.py (+112/-0); test/registered/unit/layers/quantization/test_mxfp4_situ_weight_layout.py (+37/-0)
LABELS: amd, run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ Kimi-K3 SiTUv2 MXFP4 MoE requires different preshuffled weight layouts for different activation quantization modes: ⏎  ⏎ - A16W4 and A8W4 use AITER's GU-interleaved `shuffle_weight_a16w4` layout. ⏎ - A4W4 uses the generic separated layout. ⏎ - When both A8W4 and A4W4 flags are set, AITER gives A8W4 precedence. ⏎  ⏎  ⏎ ### AITER dependency ⏎  ⏎ This PR requires [ROCm/aiter#4534](https://github.com/ROCm/aiter/pull/4534) (`6dc26b7a817bba2ae9 …[truncated]

### L1-3bac084d4e  (L1, 2026-09-03, sha 3bac084d4e49, PR #37654)
TITLE: [Model] Add native IFM K2 Horizon serving support (#37654)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/__init__.py (+3/-0); python/sglang/srt/configs/k2_horizon.py (+30/-0); python/sglang/srt/configs/model_config.py (+1/-0); python/sglang/srt/constrained/grammar_manager.py (+20/-7); python/sglang/srt/constrained/reasoner_grammar_backend.py (+22/-2); python/sglang/srt/disaggregation/decode.py (+10/-0); python/sglang/srt/entrypoints/openai/protocol.py (+1/-0); python/sglang/srt/entrypoints/openai/serving_chat.py (+31/-0); python/sglang/srt/entrypoints/openai/serving_responses.py (+17/-0); python/sglang/srt/function_call/function_call_parser.py (+2/-0); (+24 more)
LABELS: run-ci, bypass-fastfail, run-ci-extra
BODY: > Draft for public review. The K2 Horizon model repositories remain private until release, so please do not merge this PR before the release gate is cleared. ⏎  ⏎ ## Motivation ⏎  ⏎ Add native SGLang serving support for the IFM K2 Horizon family without checkpoint-provided Python code. ⏎  ⏎ ## Modifications ⏎  ⏎ - Add native xLLM/K2 Horizon dense, MoE, and MoVA model support. ⏎ - Add K2 Horizon reasoning and tool-call parsers with template detection. ⏎ - Support non …[truncated]

### L1-397aeca376  (L1, 2026-09-03, sha 397aeca37624, PR #37623)
TITLE: fix(benchmark): support Glm4MoeLite in fused MoE tuner (#37623)
SOURCES: subject_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: benchmark/kernels/fused_moe_triton/common_utils.py (+2/-0)
LABELS: run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎ The fused-MoE tuning utility does not classify ⏎ `Glm4MoeLiteForCausalLM` as a DeepSeek-style architecture. As a result, it ⏎ cannot derive the routed/shared-expert dimensions used by GLM-4.7-Flash. ⏎  ⏎ ## Modifications ⏎  ⏎ - Add `Glm4MoeLiteForCausalLM` to the DeepSeek-style fused-MoE configuration ⏎   path. ⏎ - Include its shared expert when fusion is enabled, matching the model's ⏎   serving layout. ⏎  ⏎ For GLM-4.7-Flash at TP1, the tuner now deri …[truncated]

### L1-3ad3f23ed5  (L1, 2026-09-03, sha 3ad3f23ed590, PR #37206)
TITLE: [Comm] Drop the in-tree MNNVL CuTe DSL port in favor of FlashInfer 0.6.18 (#37206)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/communication/mnnvl_cutedsl/__init__.py (+0/-52); python/sglang/kernels/ops/communication/mnnvl_cutedsl/config.py (+0/-231); python/sglang/kernels/ops/communication/mnnvl_cutedsl/cute_dsl_primitives.py (+0/-876); python/sglang/kernels/ops/communication/mnnvl_cutedsl/kernel_bt/__init__.py (+0/-43); python/sglang/kernels/ops/communication/mnnvl_cutedsl/kernel_bt/device_kernels.py (+0/-1285); python/sglang/kernels/ops/communication/mnnvl_cutedsl/kernel_bt/protocol.py (+0/-511); python/sglang/kernels/ops/communication/mnnvl_cutedsl/kernel_ht/__init__.py (+0/-37); python/sglang/kernels/ops/communication/mnnvl_cutedsl/kernel_ht/device_kernel.py (+0/-978); python/sglang/kernels/ops/communication/mnnvl_cutedsl/kernel_ht/protocol.py (+0/-469); python/sglang/kernels/ops/communication/mnnvl_cutedsl/kernel_ll/__init__.py (+0/-47); (+7 more)
LABELS: run-ci, jit-kernel
BODY: ## Summary ⏎  ⏎ FlashInfer 0.6.18 ships `flashinfer.comm.mnnvl_cutedsl` and exposes the backend through the public `allreduce_fusion` API, so this removes the vendored copy under `sglang.kernels.ops.communication` (byte-identical to the released modules apart from import spelling) and imports the released backend directly. ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :x: [Run #33575884763](https://github.com/sgl-project/sglang/actions/runs/3357588476 …[truncated]
