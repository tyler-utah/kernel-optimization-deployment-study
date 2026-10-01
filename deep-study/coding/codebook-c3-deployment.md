# Codebook C2/C3/C4 — evaluation audit and deployment-evidence ladder (v1.0)

Unit: one paper classified kernel-style `yes` (all positives) — or a sampled
negative for the missed-adoption check (then only C3 fields are required).

Dates: MLSys 2025 was held 12–15 May 2025; ASPLOS 2025 was 30 Mar–3 Apr
2025. Study cutoff: 2026-09-28. Observations after the cutoff do not count.

## C2 — evaluation audit (from the paper text)

| Field | Values |
|---|---|
| `eval_scope` | `microbenchmark_only` (kernel/operator level only), `end_to_end_only`, `both` |
| `baselines` | named baselines with versions if stated, e.g. `vLLM v0.5.3; FlashInfer 0.1.6; cuBLAS 12.4` |
| `baseline_versions_stated` | `yes` / `partial` / `no` |
| `n_hardware_targets` | integer number of distinct hardware platforms evaluated (GPU models, accelerators, CPUs) |
| `hardware_targets` | `;`-separated, e.g. `A100;H100` |
| `hardware_vendors` | `;`-separated, e.g. `NVIDIA;AMD` |
| `correctness_validation` | `none_reported`, `tolerance_vs_reference` (numerical comparison), `model_accuracy_eval` (end-task accuracy/perplexity), `both`, `formal_or_exact` (proof/bit-exact) |
| `public_code` | `yes` / `no` |
| `code_url` | repository URL named in the paper (or found as the official artifact) |
| `artifact_badges` | `;`-separated among `available`, `functional`, `reusable`, `reproduced`; `none_found` if the paper shows none; `unknown` if full text was unavailable |
| `claimed_production` | `yes` if the paper states deployment in production (level S), else `no` |
| `claimed_production_quote` | ≤200-char quote, or "" |

## C3 — deployment ladder (highest level with observable public evidence)

| Level | Evidence required (every L3–L5 label needs URL + quote) |
|---|---|
| `L0` | no public code |
| `L1` | code released |
| `L2` | code maintained after publication: commits to the official repo **more than 3 months after the conference**, or issues answered then, or a released pip package with a post-conference release |
| `L3` | technique/implementation picked up by a kernel library (FlashInfer, xFormers, Liger-Kernel, AITER, CUTLASS, TileLang, ThunderKittens, FBGEMM, DeepGEMM…) |
| `L4` | merged into a serving or training framework (vLLM, SGLang, TensorRT-LLM, llama.cpp, PyTorch, Megatron, DeepSpeed) — code, not only an open PR/issue |
| `L5` | on by default, documented in framework docs, or in framework release notes |

`S` (self-reported production) is recorded separately in `claimed_production`.
Record the highest level reached (`ladder_level`) **and** each level's
evidence. Open-but-unmerged PRs/issues count only as `in_progress_signal`.
"The paper's own repo runs on vLLM" (a fork/plugin) is L1/L2, not L4.

Required searches (record every query in `search_log`): exact title; technique
/system name (+ distinctive kernel names); arXiv id; official artifact repo
name; author GitHub handles where known; "adapted from"/"based on"/
"inspired by" comments. Search locations:

- Local shallow clones (grep, case-insensitive): vLLM and SGLang at the frozen
  SHAs in `experiments/deep-study/data/src/{vllm,sglang}`, and downstream HEADs
  in `experiments/deep-study/data/src/downstream/<owner>__<repo>` (FlashInfer,
  xFormers, AITER, llama.cpp, Megatron-LM, DeepSpeed, TensorRT-LLM, PyTorch,
  Liger-Kernel, CUTLASS);
- GitHub PR/issue search: `gh search prs "<term>" --repo <owner/repo>` and
  `gh search issues …` (≤30 req/min; `gh search code` ≤10 req/min).
- web_search for docs/release notes when a framework mention is found.

Common technique names can produce false hits (e.g. "stream-K" existed
before the paper). A hit counts only if it plausibly refers to *this paper's*
technique/implementation (citation, link, name + mechanism match, or authors).

## Fields

`id, eval_scope, baselines, baseline_versions_stated, n_hardware_targets,
hardware_targets, hardware_vendors, correctness_validation, public_code,
code_url, artifact_badges, claimed_production, claimed_production_quote,
ladder_level, l2_evidence, l3_evidence, l4_evidence, l5_evidence,
in_progress_signal, preprint_date, first_code_date, first_downstream_pr_date,
downstream_merge_date, first_default_or_doc_date, search_log, confidence, notes`

- Each `lN_evidence` is `""` or `"<URL> | <≤200-char quote>"` (use `;;` to separate multiple).
- Dates `YYYY-MM-DD` or `""`; `preprint_date` = first arXiv version if any.
- `search_log`: list of strings `"<where>: <query> -> <n hits / verdict>"`.
