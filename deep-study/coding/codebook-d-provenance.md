# Codebook D — reverse provenance of deployed kernel/backend families (v1.0)

Unit: one **kernel family implementation/backend** present in a repository at
the frozen HEAD (vLLM `7230dfea501b`, SGLang `79cafec013d0`). A family is a
selectable or dispatched implementation of one operator class, e.g. "FlashInfer
attention backend", "Triton fused-MoE", "Marlin W4A16 GEMM", "custom
all-reduce". Different backends for the same operator are separate records.

Required operator classes (enumerate every implementation you can verify in
each): `attention` (incl. MLA / sparse / linear-attention kernels),
`fused_moe`, `quantized_gemm` (quantized linear / scaled-mm / grouped GEMM),
`norm_activation` (RMSNorm, fused add-norm, activation-and-mul, rotary),
`sampling` (top-k/top-p, penalties, rejection sampling), `speculative_decoding`
(tree/verification/draft kernels), `collective` (custom all-reduce, all-to-all,
fused comm+compute), and `other_kernel` (e.g. KV-cache copy/quant, Mamba/SSM,
LoRA punica) when clearly kernel-level.

## Columns (CSV, UTF-8, header row exactly as below)

`repo,operator_class,family,implementation_name,source_paths,supported_hardware,upstream_or_vendored_source,cited_paper_or_system,provenance_class,provenance_evidence,evidence_urls,first_intro_ref,first_intro_date,status,status_evidence,confidence,notes`

- `source_paths`: `;`-separated repo-relative paths at the frozen HEAD (verify they exist in `experiments/deep-study/data/src/<repo>`).
- `supported_hardware`: e.g. `NVIDIA sm80+`, `NVIDIA sm90`, `AMD MI300`, `Intel XPU`, `TPU`, `Ascend NPU`, `CPU`.
- `upstream_or_vendored_source`: pip dependency, git submodule/CMake FetchContent, vendored directory, or `in-repo`. Name the project (e.g. `flashinfer-python`, `vllm-project/flash-attention fork of Dao-AILab/flash-attention`, `deepseek-ai/DeepGEMM`, `ROCm/aiter`).
- `cited_paper_or_system`: paper title/arXiv id or named system cited in code comments, docs, or introducing PR; `none found` otherwise.
- `provenance_class` (origin of the *technique/implementation*, not of the integration glue):
  `academic_paper` (implementation originates from or directly implements a published academic paper/its artifact, e.g. FlashAttention, FlashInfer, PagedAttention, Marlin, QServe),
  `vendor_library` (hardware vendor library or vendor-authored kernels: CUTLASS, cuBLAS, TensorRT-LLM kernels, AITER, oneDNN/IPEX, NPU vendor ops),
  `company_engineering` (non-vendor company kernels released without an academic paper, e.g. DeepSeek DeepGEMM/FlashMLA/DeepEP, Meta/xFormers-internal kernels),
  `community_contribution` (written in-repo by project contributors without an upstream paper/vendor/company source),
  `unclear`.
  If a family has a paper *and* vendor/company authorship, choose the class of the primary origin and explain in `notes`.
- `provenance_evidence`: ≤250-char verbatim quote (code comment, doc line, README, PR text) supporting the class.
- `evidence_urls`: `;`-separated URLs (GitHub blob URLs at the frozen SHA, PRs, papers, docs).
- `first_intro_ref`: introducing PR (`#NNNN`) or commit SHA (first 12 chars) that first added the family to this repo; find via `git -C experiments/data/repos/<repo> log --first-parent --diff-filter=A --no-renames --format="%H %as %s" -- <path>` (ALWAYS pass `--no-renames`; the clone is blobless and rename detection triggers slow network fetches) or `gh`/GitHub search. `unknown` if not traceable.
- `first_intro_date`: YYYY-MM-DD.
- `status`: `default` (used by default on at least one mainstream hardware/config), `selectable` (user/config selectable or auto-selected only in special cases), `experimental` (flagged experimental/env-var gated/prototype), `deprecated`.
- `status_evidence`: short quote/path showing default/selection logic.
- `confidence`: high/medium/low.

Rules: verify every claim from code, history, docs or introducing PRs; do not
rely only on arXiv URLs. Never invent a PR number or paper. Do not name or rank
individual contributors.
