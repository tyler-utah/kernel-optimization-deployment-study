# Codebook C1 — kernel-style optimization census (v1.0)

Unit: one research paper in the official MLSys 2025 or ASPLOS 2025 proceedings.
Input: title, authors, abstract, extracted introduction/contributions,
evaluation excerpt, conclusion excerpt, code links, artifact-badge text,
production-claim snippets, baseline/version snippets and hardware mentions.
Full text is in `experiments/deep-study/data/cache/papers/text/<id>.txt` when
`fulltext_available` is true — open it whenever the excerpts are insufficient
(e.g. contributions list or evaluation summary missing).

## Construct (broad definition from studies/04-paper-to-production.md)

A paper **proposes a kernel-style optimization** (`kernel_style=yes`) when a
primary contribution is at least one of:

- `kernel` — a new or improved compute kernel (attention variant, GEMM, MoE,
  sampling, normalization, quantized matmul, collectives, sparse kernels,
  convolution, scan …) on GPUs/accelerators/CPUs;
- `fusion_family` — operator fusion, mega-kernels, persistent kernels,
  fused communication+computation kernels;
- `precision_format` — a numeric precision/format change that needs kernel
  support (FP8, FP4, INT4, KV-cache quantization, mixed precision kernels);
- `scheduling_layout` — memory-layout or scheduling change implemented at
  kernel level (paged/ragged layouts, split-K, stream-K, tiling, work
  partitioning, intra-kernel pipelining, warp specialization);
- `compiler_dsl` — compiler/DSL/autotuner/superoptimizer work whose output is
  kernels (Triton/TileLang-style DSLs, tensor compilers, auto-scheduling);
- `agentic_generation` — LLM/agent-driven kernel generation or optimization.

`kernel_style=no` (`category=not_kernel_style`) for work whose contributions
live above the kernel: cluster/request scheduling, serving system policies,
parallelism strategy search without kernel generation, memory management
above kernels (unless kernel-level), training algorithms, model compression
without kernels, hardware architecture proposals evaluated only in simulation
(unless they also contribute kernels/compilers for real devices), OS,
security, storage, networking, verification, etc.

Hardware-architecture papers (common at ASPLOS): a new accelerator/ISA/
microarchitecture evaluated in simulation is `no` unless it also contributes a
kernel/compiler/dataflow mapping that is itself the main contribution and is
software-realizable on existing hardware — then `ambiguous`, explained.

`ambiguous` = genuinely borderline (e.g. system paper whose key mechanism is a
kernel but the paper frames it as system design). Explain precisely.

## Fields

| Field | Values |
|---|---|
| `id` | copy |
| `kernel_style` | `yes` / `no` / `ambiguous` |
| `category` | primary: `kernel`, `fusion_family`, `precision_format`, `scheduling_layout`, `compiler_dsl`, `agentic_generation`, `not_kernel_style` (must be `not_kernel_style` iff `kernel_style=no`) |
| `secondary_categories` | list of other applicable categories |
| `technique_name` | the system/technique name used in the paper (e.g. "FlashInfer"), or "" |
| `target_workload` | `llm_inference`, `llm_training`, `other_dnn`, `non_ml`, `mixed` |
| `rationale` | 1–3 sentences grounded in the paper's stated contributions |
| `confidence` | `high` / `medium` / `low` |
| `sections_read` | list from `abstract`, `introduction`, `contributions`, `evaluation`, `conclusion`, `fulltext` |
