### L3-4be061c5af  (L3, 2026-09-26, sha 4be061c5afab, PR #58846)
TITLE: [Kernel] Bump FlashKDA to keep the recurrent state in fp32 (#58846)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: cmake/external_projects/flashkda.cmake (+1/-1)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ Bumps the FlashKDA pin from `b59532f` to `17a037d`, the merge of vllm-project/FlashKDA#13 on its `dev` branch. ⏎  ⏎ Before this change, FlashKDA's K2 kernel stored the KDA recurrent state in bf16 and rounded it after every 16-token tile. On long GLM-5.3-Flash prefills the error builds up across tiles and across chunked-prefill boundaries, and it shows up as corrupted long-context tool-call output. FlashKDA#13 keeps the state in fp32 b …[truncated]

### L3-3137ff0773  (L3, 2026-09-27, sha 3137ff07739f, PR #58201)
TITLE: [ROCm][Kimi-K3] Make VLLM_ROCM_USE_AITER_MOE_SITUV2 select a4w4/a8w4/a16w4 (#58201)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/kernels/moe/test_rocm_aiter_moe.py (+38/-46); vllm/_aiter_ops.py (+51/-23); vllm/envs.py (+8/-13); vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py (+5/-4); vllm/model_executor/layers/fused_moe/oracle/mxfp4.py (+1/-2)
LABELS: rocm, ready, kimi, k3
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Purpose ⏎  ⏎ The SiTU MXFP4 MoE has three FlyDSL kernel families that differ only in activation precision; the vLLM ⏎ flag used to be a bool (on = a4w4, off = a16w4) with no way to pick a8w4. Now it takes the dtype name so ⏎ the three can be A/B'd on the same server config. ⏎  ⏎ K3 MoE activation option — `VLLM_ROCM_USE_AITER_MOE_SITUV2=a16w4|a8w4|a4w4` ⏎  ⏎ | value | AITER env vLLM exports | stage-1 kernel family | tuned CSV (aiter) | ⏎ |---|---|---|- …[truncated]

### L3-77871126f9  (L3, 2026-09-27, sha 77871126f9b6, PR #58586)
TITLE: [Kernel][DSV4.1] Fuse MoE finalize into the TP all-reduce + mHC boundary (#58586)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+0/-2); csrc/libtorch_stable/all_reduce_mhc.cu (+0/-212); tests/distributed/test_custom_all_reduce.py (+120/-73); vllm/models/deepseek_v41/nvidia/model.py (+76/-4); vllm/models/deepseek_v41/nvidia/ops/cute_dsl/__init__.py (+8/-0); vllm/models/deepseek_v41/nvidia/ops/cute_dsl/all_reduce_mhc.py (+1134/-0); vllm/models/deepseek_v41/nvidia/ops/cute_dsl/primitives.py (+396/-0); vllm/models/deepseek_v41/nvidia/ops/mhc.py (+73/-18)
LABELS: ready, ci/build, deepseek, nvidia, quantization, DSv4.1
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ DSV4.1 decode (TP4, `0 < M <= 16`, FULL CUDA graphs) closes every sublayer with a fused TP all-reduce + mHC kernel. After an MoE, that boundary is preceded by FlashInfer's MoE finalize (top-k weighted sum) and a separate shared-expert add. This PR folds the finalize and the shared add into the boundary: ⏎  ⏎ - **Vendored FlashInfer LL kernels.** `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/` copies FlashInfer's CuTe DSL low-latency MNNVL …[truncated]

### L3-24c9772d19  (L3, 2026-09-27, sha 24c9772d1925, PR #54967)
TITLE: [Attention][CPU] Use zentorch SDPA for CPU MLA prefill (#54967)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/prefill/registry.py (+4/-0); vllm/v1/attention/backends/mla/prefill/selector.py (+12/-2); vllm/v1/attention/backends/mla/prefill/zen_cpu_sdpa.py (+106/-0); tests/v1/attention/test_cpu_mla_backend.py (+67/-1); tests/v1/attention/test_mla_prefill_selector.py (+33/-0)
LABELS: ready, cpu
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: - Route per-request prefill attention in CPUSDPAMLAPrefillBackend through zentorch_sdpa on AMD Zen CPUs when zentorch is available, falling back to the reference matmul/softmax path ⏎  ⏎  ⏎  ⏎ ## Test Result ⏎  ⏎ tests/v1/attention/test_cpu_mla_backend.py: All 4 tests passed ⏎  ⏎ test_kv_cache_cpu_write PASSED ⏎ test_cpu_mla_prefill_backend_selected PASSED ⏎ test_cpu_mla_prefill_new_tokens PASSED ⏎ test_cpu_mla_prefill_context_chunk PASSED ⏎  ⏎  ⏎ ## Performanc …[truncated]

### L3-c8d7a7dd13  (L3, 2026-09-27, sha c8d7a7dd13e2, PR #58684)
TITLE: [Perf][Attention] Remove D2H sync from FlashInfer SM90 sparse MLA plan under async scheduling (#58684)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.flashinfer_sparse
FILES: vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py (+200/-78); tests/v1/attention/test_flashinfer_mla_sparse_sm90.py (+181/-28)
LABELS: nvidia
DEEP_STUDY: deep-study performance PR (perf_regression_fix)
BODY: ## Purpose ⏎  ⏎ With async scheduling, `FLASHINFER_MLA_SPARSE_SM90` did a blocking D2H copy on every metadata build. FlashInfer bakes each row's `kv_len` into its schedule on the host, and `seq_lens_cpu_upper_bound` is optimistic under async spec decode, so the builder copied `positions` / `seq_lens` back to get exact lengths. With MTP that happens on every draft step too. On GLM-5.3-Flash this made async scheduling *slower* than `--no-async-scheduli …[truncated]

### L3-0376f81530  (L3, 2026-09-27, sha 0376f8153033, PR #57214)
TITLE: [Perf][Pooling] Avoid blocking seq_lens GPU-to-CPU copy for pooling in FlashInfer metadata builder (#57214)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+10/-2); tests/v1/attention/test_attention_backends.py (+87/-1)
LABELS: performance, ready, nvidia, mrv2, verified
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ FlashInfer's metadata builder needs per-request sequence lengths on the CPU to plan paged KV metadata. For V2 model runner, `seq_lens` live on the GPU, so the builder fetches them with `common_attn_metadata.seq_lens.cpu()`. For pooling models this copy is avoidable; pooling models have no speculative tokens, so `seq_lens_cpu_upper_bound` maintained by the model runner is exact, not optimistic. The pooling runner already relies on this …[truncated]

### L3-231fdb83cc  (L3, 2026-09-27, sha 231fdb83cccf, PR #58880)
TITLE: [Perf][MoE] Use fused MiniMax2 routing with non-unit routed scaling (#58880)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_flashinfer.py (+97/-0); tests/kernels/moe/test_routing.py (+32/-0); vllm/model_executor/layers/fused_moe/config.py (+2/-5); vllm/model_executor/layers/fused_moe/router/cpu_router.py (+0/-1); vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py (+0/-1); vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py (+0/-1); vllm/model_executor/layers/fused_moe/router/router_factory.py (+0/-1); vllm/model_executor/layers/fused_moe/router/zero_expert_router.py (+4/-12)
LABELS: ready, cpu, nvidia, minimax
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ `get_routing_method_type` maps sigmoid + correction bias + renormalize routing (no expert groups) to `RoutingMethodType.MiniMax2` only when `routed_scaling_factor` is 1.0 (added in #44347). MiniMax-M3 uses `routed_scaling_factor=2.0`, so its MoE falls back to `Unspecified`. On SM100 with the MXFP8 checkpoint, that means the modular TRT-LLM path: a separate vLLM `topk_sigmoid` launch per MoE layer, then `trtllm_fp8_block_scale_routed_m …[truncated]

### L3-55de40a2fc  (L3, 2026-09-28, sha 55de40a2fc1a, PR #58673)
TITLE: [Minimax-M3] Add Encoder CUDA graph support (#58673)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/multimodal/generation/test_vit_cudagraph.py (+18/-0); vllm/models/minimax_m3/amd/model.py (+8/-1); vllm/models/minimax_m3/common/encoder_cudagraph.py (+281/-0); vllm/models/minimax_m3/common/vision_tower.py (+82/-15); vllm/models/minimax_m3/nvidia/model.py (+8/-1)
LABELS: ready, ci/build, multi-modality, nvidia, minimax
BODY: ## Purpose ⏎  ⏎ Add encoder CUDA graph support for Minimax-M3 ⏎  ⏎ ### E2E perf ⏎  ⏎ Load: random-mm ⏎  ⏎ [details omitted] ⏎  ⏎ #### GB300, TP4 ⏎  ⏎ [details omitted] ⏎  ⏎ Concurrency | Main median latency | Feature median latency ⏎ -- | -- | -- ⏎ 1 | 199–211 ms | 178–179 ms ⏎ 4 | 347–598 ms | 353–527 ms ⏎ 16 | 457–464 ms | 398–434 ms ⏎  ⏎ #### MI355X, TP4 ⏎  ⏎ [details omitted] ⏎  ⏎ Concurrency | Main @ 5ff3bbfb08 | PR (encoder CG) | Δ ⏎ -- | -- | -- | -- ⏎ 1 | 102.2 ms …[truncated]

### L3-af7f9488c2  (L3, 2026-09-28, sha af7f9488c221, PR #54628)
TITLE: [Benchmark] Add Responses API backend to vllm bench serve (#54628)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/benchmarking/cli.md (+40/-0); tests/benchmarks/test_responses_request_func.py (+455/-0); vllm/benchmarks/datasets/datasets.py (+3/-1); vllm/benchmarks/lib/endpoint_request_func.py (+179/-1); vllm/benchmarks/serve.py (+5/-0)
LABELS: documentation, performance, ready, verified
BODY: ## Purpose ⏎  ⏎ `vllm bench serve` cannot benchmark `/v1/responses`. `ASYNC_REQUEST_FUNCS` covers completions, chat, audio, embeddings, pooling and rerank, and there is an explicit `# TODO: Add more request functions for different API protocols.` right above it, but no Responses request function. This adds an `openai-responses` backend so the Responses protocol can be measured directly instead of routing the same workload through `/v1/chat/completion …[truncated]

### L3-6b24bd8e75  (L3, 2026-09-28, sha 6b24bd8e75a3, PR #58208)
TITLE: [ROCm][Perf] Replace torch.topk in DSA candidate block selection (#58208)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/kernels/attention/dsa/candidate_blocks.py (+256/-0)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ `select_candidate_blocks` picks the top-k blocks per row with `torch.topk`, which runs a multi-pass radix select (`computeBlockDigitCounts`, `computeBlockwiseWithinKCounts`, `gatherTopK`). That does two things this caller does not need: ⏎  ⏎ - **It orders the results.** The only consumer is `_candidate_flags_kernel`, which does `flags[block] = 1`. Nothing downstream reads the ranking or the scores — only the *set* of selected block id …[truncated]

### L3-94d1462924  (L3, 2026-09-28, sha 94d146292479, PR #58814)
TITLE: [Bugfix][Kimi-K3] Refresh DSpark context KV cache pointers after the KV cache is re-bound (#58814)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/kimi_k3/nvidia/dspark_mla.py (+6/-1)
LABELS: bug, ready, dflash, kimi, k3
BODY: ##Purpose ⏎  ⏎ Fixes a GPU memory fault and silent memory corruption in the Kimi-K3 DSpark draft, introduced by #57632. ⏎  ⏎ ``` ⏎ 21:05:36  JIT kernel warmup finished            (all 8 ranks) ⏎           Warning: Queue error - HSA_STATUS_ERROR_MEMORY_FAULT ⏎           Memory access fault by GPU node-14 ... on address 0x7ec3c95b0000. Reason: Unknown. ⏎ 21:15:38  [Rank 0..7] Watchdog caught collective operation timeout: _ALLGATHER_BASE ... 600000 ms ⏎      …[truncated]

### L3-764413559a  (L3, 2026-09-28, sha 764413559a14, PR #58498)
TITLE: [Bugfix] Don't sync-police or retry FlashInfer all-reduce workspace creation in eager mode (#58498)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/distributed/test_comm_ops.py (+30/-0); vllm/distributed/device_communicators/flashinfer_all_reduce.py (+26/-14)
LABELS: bug, nvidia
BODY: ## Purpose ⏎  ⏎ Fix a nightly regression from #58197 ("[Core] Disable JIT warmup in eager mode"). ⏎  ⏎ With `--enforce-eager` the V2 `warmup_kernels` dummy forward is now skipped. The standalone FlashInfer all-reduce (`FlashInferAllReduce`, enabled by default through `VLLM_ALLREDUCE_USE_FLASHINFER=1`) creates its workspace lazily. The profile run uses large tensors that are above the FlashInfer size cap (e.g. 0.5 MB at TP8 or 2 MB at TP4 on H200), so it  …[truncated]

### L3-4694145045  (L3, 2026-09-28, sha 469414504538, PR #58646)
TITLE: [Skills] Update kernel-microbenchmark to include ROCm (#58646)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .agents/skills/kernel-microbenchmark/SKILL.md (+23/-2); .agents/skills/kernel-microbenchmark/benchmarks/graph_replay_benchmark.py (+107/-0)
LABELS: rocm
BODY: ## Purpose ⏎  ⏎ Update `kernel-microbenchmark` skill to include ROCm ⏎ - There is no flashinfer cupti, hence use HIP graph replay with rotating buffers exceeding LLC size. ⏎ - This is better than Triton `do_bench` because Triton `do_bench` still has significant launch overheads. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-004e37ece2  (L3, 2026-09-28, sha 004e37ece2fd, PR #58557)
TITLE: [Misc] Name each backend and its kernel block sizes in block-size errors (#58557)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backend.py (+3/-0); tests/v1/worker/test_attn_utils.py (+1/-0); tests/v1/worker/test_gpu_model_runner.py (+4/-0); vllm/v1/worker/utils.py (+7/-1)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ `select_common_block_size` raises `No common block size for N.` without saying which attention backends disagreed or what they support, which is the opaque error behind #48286. This names each backend and its supported kernel block sizes: ⏎  ⏎ ``` ⏎ main: No common block size for 96. ⏎ PR:   No common block size for 96 (A: [MultipleOf(48)]; B: [64]). ⏎ ``` ⏎  ⏎ `MultipleOf` also had no `__repr__`, so any message that formats `get_supported_kernel_b …[truncated]

### L3-25624f65cf  (L3, 2026-09-28, sha 25624f65cf40, PR #56861)
TITLE: [ROCm][MLA] Add an AITER ASM round-robin decode route for DCP multi-token verify (#56861)
SOURCES: path_core, path_integration+keyword, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.mla.rocm_aiter
FILES: vllm/envs.py (+10/-0); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+258/-23); tests/v1/attention/test_rocm_aiter_mla_dcp_cprr.py (+152/-0); tests/v1/attention/test_rocm_aiter_mla_dcp_cprr_numerics.py (+540/-0); tests/v1/attention/test_rocm_aiter_mla_mtp_split.py (+242/-7)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ Adds an opt-in AITER ASM route for decode context parallelism (DCP) combined with speculative decoding on ROCm gfx950. ⏎  ⏎ With DCP, each rank holds a round-robin shard of the KV cache (token `t` on rank `t % dcp_world_size`). A causal verify step of `qlen > 1` cannot run on the ordinary decode kernel, because that kernel applies causality on local indices, which is wrong over a round-robin shard. Today these steps are served by the segm …[truncated]

### L3-6c300ddb9e  (L3, 2026-09-28, sha 6c300ddb9e9d, PR #59008)
TITLE: [CI][Bugfix] Relax packed_qk_rope_ correctness test to one ULP (#59008)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/core/test_apply_rotary_emb.py (+18/-5); vllm/model_executor/layers/rotary_embedding/packed_qk_rope.py (+4/-2)
LABELS: bug, rocm, ready
BODY: ## Purpose ⏎  ⏎ `tests/kernels/core/test_apply_rotary_emb.py::test_packed_qk_rope_correctness` asserts that `packed_qk_rope_` is bitwise identical to `ApplyRotaryEmb(enable_fp32_compute=True)`. That fails on ROCm: 157 of 216 shape/dtype/seed combinations disagree, always by exactly one ULP. ⏎  ⏎ **Why**  ⏎ Every mismatch sits at an odd `head_dim` index, i.e. only in `o1`.`o0` reads the same four values and is exact everywhere, so both sides seebitwise …[truncated]

### L3-75fad5bbef  (L3, 2026-09-28, sha 75fad5bbef7b, PR #58058)
TITLE: [Bugfix][ROCm] Drop -1 sentinels when building the ragged sparse-MLA indices (#58058)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+92/-27); tests/kernels/attention/test_rocm_triton_attn_dsv4.py (+32/-0)
LABELS: bug, rocm, ready
BODY: ## Purpose ⏎ The sparse MLA kernel (pa_sparse_prefill_opus) currently seg. faults when -1 sentinels happen within accessed region as there is no check for it. ⏎  ⏎  `combine_topk_swa_indices` reserves top-k columns from a position-derived count and writes `-1` wherever a candidate fails its validity test, and `build_ragged_indices_from_dense` copies the row prefix verbatim, sentinels included. ⏎   ⏎  ## The fix ⏎  ⏎ Make the ragged index stream sentinel …[truncated]

### L3-6d3ea3c2c9  (L3, 2026-09-28, sha 6d3ea3c2c903, PR #50499)
TITLE: [KVConnector][NIXL] Support packed MLA KV layouts in pipeline-parallel push prefill (#50499)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: docs/design/nixl_kv_push_connector.md (+29/-11); docs/features/nixl_connector_compatibility.md (+2/-1); tests/v1/kv_connector/unit/test_nixl_desc_geometry.py (+203/-2); tests/v1/kv_connector/unit/test_nixl_push_connector.py (+36/-0); vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py (+83/-13); vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py (+4/-1)
LABELS: documentation, ready, kv-connector
BODY: ## Purpose ⏎  ⏎ Extend merged #50494 to packed MLA KV caches (block-outermost layouts, e.g. DeepSeek-V4-Flash) in pipeline-parallel `NixlPushConnector` prefill. A PP stage and the decoder pack different layers at different offsets and strides, so the producer must address individual layer pages rather than copy whole packed rows. ⏎  ⏎ - PP producers register one logical transfer region per layer; each physical allocation is still registered once. ⏎ - PP=1  …[truncated]

### L3-d6cce94fd4  (L3, 2026-09-28, sha d6cce94fd422, PR #58957)
TITLE: [Perf][Qwen4Exp] Fuse HC down projection and SiLU on NVIDIA (#58957)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/qwen4_exp/test_hc_ops.py (+30/-0); vllm/models/qwen4_exp/nvidia/hyperconnection.py (+38/-23); vllm/models/qwen4_exp/nvidia/model.py (+11/-0); vllm/models/qwen4_exp/nvidia/ops/cute_dsl/__init__.py (+2/-0); vllm/models/qwen4_exp/nvidia/ops/cute_dsl/_hc_down_silu_fma.py (+334/-0); vllm/models/qwen4_exp/nvidia/ops/cute_dsl/_hc_down_silu_mma.py (+536/-0); vllm/models/qwen4_exp/nvidia/ops/cute_dsl/hc_down_silu.py (+224/-0)
LABELS: qwen, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Fuse Qwen4Exp's BF16 HC down projection and SiLU for NVIDIA decode batches up to 48 tokens, with the existing unfused path for larger batches. This is separate from #58706 (ROCm) and #53909 (other HC operations). ⏎  ⏎ The CuTe DSL kernel adapts vLLM's `ll_bf16` dot-product and split-K GEMM paths. It applies SiLU in the output epilogue, preserves the `ll_bf16` BF16 rounding boundary, and skips the 12 unused padding columns. ⏎  ⏎ ## Microbenchm …[truncated]

### L3-20b52e9f5b  (L3, 2026-09-28, sha 20b52e9f5b57, PR #58114)
TITLE: [Perf][Qwen3.8] Reduce PLE metadata construction overhead (#58114)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/qwen4_exp/test_ple.py (+125/-0); vllm/models/qwen4_exp/amd/ple_layer.py (+2/-7); vllm/models/qwen4_exp/nvidia/ple_layer.py (+1/-2); vllm/v1/attention/backends/mamba_attn.py (+11/-9); vllm/v1/attention/backends/short_conv_attn.py (+22/-60)
LABELS: ready, qwen
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ PLE metadata construction still synchronizes in mixed speculative batches and builds temporary tensors that its consumers do not need. This reduces per-step builder work while preserving request ordering, accepted-token routing, and persistent CUDA graph buffers. ⏎  ⏎ - Supply `repeat_interleave` with the output size from CPU query offsets, excluding token padding, and use `index_fill_` for request-group labels to avoid scalar-assignment  …[truncated]

### L3-522101f6ed  (L3, 2026-09-28, sha 522101f6ede5, PR #56711)
TITLE: [Multimodal] Avoid extra d2d for encoder cudagraph with fused input norm (#56711)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/ops/vit_attn_wrappers.py (+7/-0); vllm/model_executor/layers/fusion/mm_input_norm.py (+8/-0); vllm/model_executor/models/qwen2_5_vl.py (+7/-3); vllm/model_executor/models/qwen2_vl.py (+7/-3)
LABELS: ready, qwen, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ - Follow up PR for https://github.com/vllm-project/vllm/pull/55370#issuecomment-5646031609 ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-8a497d7557  (L3, 2026-09-28, sha 8a497d75572f, PR #52395)
TITLE: [CI/Build][BugFix][The Rock] Make supports_mm_prefix  return False for ROCm attn and unified attn since Prefix-LM not implemented (#52395)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.rocm.v1_rocm_attn, L3.rocm.aiter_unified
FILES: vllm/v1/attention/backends/rocm_aiter_unified_attn.py (+2/-1); vllm/v1/attention/backends/rocm_attn.py (+2/-1)
LABELS: bug, rocm, ready
BODY: ## Purpose ⏎ This PR changes `supports_mm_prefix` in for ROCm attention and unified attention to return false since  `RocmAttentionMetadata`   ⏎ and `RocmAttentionImpl.forward` do not have `mm_prefix` (or similar) fields / parameters like the `TritonAttentionMetadata` and `TritonAttentionImpl` counteparts do so the feature is not implemented. ⏎  ⏎ This causes `Multi-Modal Models (Extended Generation 3)` to fail on` The Rock 7.14`. ⏎  ⏎ The failure aris …[truncated]
