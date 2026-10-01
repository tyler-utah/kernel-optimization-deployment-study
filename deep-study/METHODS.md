# Methods — deep empirical follow-up

This study extends the preliminary optimization-delivery study
(`experiments/METHODS.md`, `experiments/system-evolution-report.md`). It reuses
that study's frozen repository SHAs, cutoff, REST caches and first-parent logs,
and adds four work packages (A–D). All derived data, codebooks, adjudication
records and scripts are under `experiments/deep-study/`. The cross-session
checkpoint is `experiments/deep-study/STATUS.json`.

## Frozen inputs

| Item | Value |
|---|---|
| Study cutoff | 2026-09-28 23:59:59 UTC |
| vLLM HEAD | `7230dfea501b232ce92d057f1c5ee9cb009c4cb4` |
| SGLang HEAD | `79cafec013d0a01c782d7a4bc902ef32a8941ba8` |
| WP-A window (12 months) | PRs created 2025-09-29 → 2026-09-28 |
| WP-B window (24 months) | first-parent commits with committer date 2024-09-29 → 2026-09-28 |
| Downstream snapshot HEADs (WP-C search only) | recorded in `STATUS.json → downstream_snapshot_heads` (shallow clones taken 2026-09-29) |

Frozen source trees for WP-D were fetched at the two HEAD SHAs into
`data/src/{vllm,sglang}` (depth-1 fetch of the exact commit). History queries
use the first study's blobless clones with `--no-renames` (rename detection
would force blob downloads).

## Coding protocol used throughout

Contextual coding was performed by LLM subagents working from explicit,
versioned codebooks in `coding/`, over reproducible input batches in
`coding/batches/<task>/batch-NNN-input.jsonl`. Each output batch
(`…-output.jsonl`) is validated by `scripts/batches.py check <task>` (exactly
one schema-valid record per input id). Keyword prefilters were used only to
**surface candidate text**; every positive label required a coder to read the
snippet in context. Second passes were run by *fresh* subagents that were
instructed not to open first-pass outputs. Disagreements were resolved by a
third adjudicating subagent that saw both codings and the evidence
(`*_reconcile` tasks). All agreement statistics are therefore
**model–model adjudication consistency, not human inter-rater reliability**.
No person is named, profiled or ranked; contributor logins appear only inside
quoted evidence where unavoidable.

## WP-A — PR delivery

### A1 population

1. `scripts/a1_metadata.py compact` reduced the first study's complete REST
   listings (all PRs created in the window, plus the complete open queue) to
   one record per PR (25,573 vLLM and 26,004 SGLang PRs created in the window).
2. `a1_metadata.py enrich` added, for every in-window or open PR, GraphQL
   size (additions, deletions, changed files), labels, author type,
   changed paths (first 100), closing/cross-reference events and last
   comments (non-merged PRs), cached in 20–50-PR batches.
3. `a1_population.py build` computes transparent signals: performance title
   tags (`[Perf]`, `[Performance]`, `perf:` …), bug-fix and non-performance
   tags, kernel tags, the `performance` label, a title optimization-claim
   regex, production kernel paths (CUDA/HIP sources, `csrc/`, `sgl-kernel/`,
   `jit_kernel/`, fused-MoE, attention backends/ops, quantization kernels,
   device communicators, sampling/LoRA/Mamba ops, compile fusion passes;
   tests/benchmarks/docs excluded) and a body optimization-claim regex.
   Tiers: **rule_confirmed** = explicit performance tag and no fix/CI/doc/
   revert tag; **ambiguous** = any other performance or kernel signal;
   **rule_excluded** = none.
4. `a1_population.py candidates` sent to contextual adjudication (codebook
   `coding/codebook-a1-performance.md`): every in-window ambiguous
   candidate; every first-study open-queue candidate (589 vLLM, 345 SGLang);
   and seeded random validation samples of 150 rule-confirmed and 200
   rule-excluded PRs per repository. Coders did not see why a PR was selected.
5. `a1_population.py merge` writes `data/performance-pr-population.csv` and
   `data/a1-tier-validation.csv` (tier precision / miss rate with Wilson 95%
   intervals).

### A2 full-population analysis

`scripts/a2_analysis.py` uses complete metadata for every in-window PR.
Time to merge is analysed with Kaplan–Meier estimators (open PRs censored at
the cutoff; closed-unmerged PRs censored at closure) **and** Aalen–Johansen
cumulative incidence of merge with closure as a competing event. Effect sizes
(KM median difference and ratio, CIF differences at 7/30 days, 90-day
restricted mean time to merge, closed-unmerged share difference) carry
1,000-replicate nonparametric bootstrap 95% intervals, overall and within
PR-size terciles. Reverted = the PR appears as a reverted target in the
WP-B1 census; superseded = closed-unmerged PR closed by another PR/commit, a
closing cross-reference, or last-comment language (signal; confirmed only in
the detailed corpus). Nothing in A2 is causal.

### A3 detailed review corpus

`scripts/a3_details.py collect` caches, for every target PR, reviews (60) with
inline comments (30 each), issue comments (100), commits (100), files, labels
and timeline events (labels, review requests, draft/ready, force-pushes,
closures, merges, cross-references), in 10-PR GraphQL batches. Truncation is
flagged. `scripts/a3_code.py select` defines the corpus: all confirmed
performance PRs when ≤2,500 per repository, else a seeded state×month
stratified sample of 1,000; and a comparison set of ≥550 human-authored
non-performance PRs per repository matched on creation month × PR-size
tercile (tercile cut points from the performance corpus) × state.

Mechanical metrics (`a3_code.py metrics`): time to first non-author human
response (reviews, inline or issue comments; bots and CI commands excluded),
substantive review rounds (distinct UTC days with substantive non-author human
review activity), reviewers, approvals, change requests, commits after first
review and force pushes.

Content codes (`coding/codebook-a3-review.md`): 7 evidence codes and 8
reviewer-request/concern codes plus `superseded`. Dossiers present
keyword-prefiltered snippets per code with source and actor role; codes with no
snippet are negative; positives require contextual confirmation and a quoted
evidence snippet. A fresh second pass re-coded 200 random PRs per repository
(`a3_pass2`); per-code percent agreement and Cohen's κ are reported in
`data/review-coding-agreement.csv`.

### A4 rules and A5 case histories

A4 starts from the first study's 41-rule audit (`results/contribution-rules.csv`).
Mechanically observable compliance (`scripts/a4_rules.py compliance`) is
measured on the detailed corpus for both groups: leading title tag, filled
template sections (vLLM Purpose/Test Plan/Test Result; SGLang Accuracy Tests
and "Speed Tests and Profiling"/"Benchmarking and Profiling"), approval by an
author-association MEMBER/OWNER/COLLABORATOR before merge (a proxy — team and
CODEOWNERS membership are not public), `ready`/`run-ci` label applied before
merge (timeline events), test files changed, and AOT kernel changes that add
both tests and benchmarks.

Undocumented requirements: a subagent induced a 35-type reviewer-request
taxonomy from 162 performance PRs (`coding/a4-request-taxonomy.json`,
`coding/codebook-a4-requests.md`); every sampled performance PR with
substantive pre-merge review (961 PRs) was then labelled with request types and
verbatim evidence. Two independent audits of the frozen contribution documents,
PR templates, CODEOWNERS, governance docs and in-repo agent-skill guides
(`data/cache/a4/doc-files.json`) decided whether each type is written down; a
type is `no` (undocumented) only if **both** audits found nothing, `yes` only if
both agree, otherwise `partial` (`data/a4-request-documentation-final.csv`,
agreement in `data/a4-doc-check-agreement.csv`).

A5 draws, with a fixed seed, three ranked candidates per slot per repository
(two fast merges ≤2 days, two slow merges ≥60 days or ≥8 review rounds, one
reverted, one superseded/abandoned; all touching kernel paths with substantive
review) into `data/a5-case-candidates.csv`; verifying agents took rank 0 unless
the public record was too thin, confirmed every fact with `gh`, and checked
release containment with the GitHub compare API. Personal handles are removed.

### A1 recall

The rule-excluded validation sample implies performance PRs missed by the
signals; `data/a1-recall-estimate.csv` extrapolates the observed miss rate
(with Wilson bounds) to the whole excluded tier. The confirmed population is a
high-precision lower bound (estimated recall ≈ 0.61–0.66).

## WP-B — failures

**B1.** All 516 first-parent commits in the 24-month window whose subject
contains revert/rollback/back-out/undo/reapply/reland language were
adjudicated in context (commit body, revert-PR body and discussion, up to four
resolved target PRs with bodies and paths; codebook
`coding/codebook-b1-reverts.md`) → `data/confirmed-reverts.csv`.

**B2.** Frame: all 2,354 first-parent commits in the window that touch
production kernel paths and whose subject has fix/failure language (1,416
vLLM, 938 SGLang), in a seeded random order. The first 220 per repository were
screened and coded in context (PR body, linked issues, discussion, files with
+/- counts, test paths; `coding/codebook-b2-failures.md`). Stage 2 fetched the
introducing PR where the fix names one and coded what correctness evidence it
reported before merge.

**B3.** Every first-pass-confirmed case was independently re-coded
(`b3_second_pass`) for confirmation, failure class, hardware specificity and
the validation counterfactual; every disagreement was adjudicated
(`b3_reconcile`). Agreement: `data/failure-coding-agreement.csv`.

**B4.** Twelve cases were selected across classes and reconstructed from PR
timelines, diffs, issues and release containment checks.

## WP-C — papers

MLSys 2025: all 61 papers from `proceedings.mlsys.org/paper_files/paper/2025`
(official PDFs). ASPLOS 2025: the official proceedings are ACM Volume 1
(`10.1145/3669940`, 72 papers) and Volume 2 (`10.1145/3676641`, 88 papers),
identified by matching the archived ASPLOS 2025 program (Wayback snapshot of
2025-04-26; the live URL now serves ASPLOS 2026) to DOIs via Semantic Scholar
and Crossref; 24 ASPLOS '24 Volume 4 papers that were presented at the 2025
meeting are excluded. The ACM DL blocks scripted access, so ASPLOS full text
came from arXiv/open copies; for papers without a script-discoverable copy,
coders searched the web for open copies and recorded exactly which sections
they read (`sections_read`, `data/paper-census.csv`).

Classification follows `coding/codebook-c1-census.md` (broad definition of
kernel-style optimization). Every yes/ambiguous paper received a fresh second
pass; disagreements were adjudicated. Every final positive received an
evaluation audit and an L0–L5 deployment score with a logged, systematic
search (`coding/codebook-c3-deployment.md`): local grep of vLLM/SGLang at the
frozen SHAs and of shallow HEAD clones of FlashInfer, xFormers, AITER,
llama.cpp, Megatron-LM, DeepSpeed, TensorRT-LLM, PyTorch, Liger-Kernel and
CUTLASS (SHAs in `STATUS.json`); `gh search prs` across those repositories;
artifact-repository commit history; and web search for docs/release notes. A
random 20 negatives per venue received the same search. A separate **strict
verification pass** (`c3_verify`) re-checked every positive/ambiguous paper
and every negative with L3+ evidence against the level definitions (library
documentation supports L3, not L5; L4 requires a merged framework PR dated
≤ cutoff) and decided whether any L3+ negative was a missed kernel-style
paper. Verified levels replace first-pass levels; both are kept in
`data/paper-deployment-ladder.csv`.

## WP-D — provenance

Two agents enumerated kernel/backend implementations at the frozen SHAs
(`coding/codebook-d-provenance.md`), verifying paths, comments/licences,
vendored sources and introducing PRs. All 204 source paths were mechanically
checked to exist. An independent blinded second pass re-derived
`provenance_class` for every record; disagreements were adjudicated →
`data/production-kernel-provenance.csv`, `data/provenance-agreement.csv`.

## Consistency check

`scripts/reports.py` generates every report from the CSVs; each number it
prints is followed by a hidden tag
`<!-- claim:<csv>::<row filter>::<column>::<format> -->`.
`scripts/consistency.py` parses every tag, recomputes the value from the named
CSV row and compares it with the printed number; the run log is
`data/consistency-check.json`.

## Operational notes

- GraphQL collection was paced (sleep = 0.5–0.8 × query time) after GitHub's
  secondary CPU-time limit killed the first unpaced 6-way run; heavy
  enrichment queries were reduced to 20 PRs to stay under the server timeout.
- A mistaken `git log` without `--no-renames` on the first study's blobless
  clone lazily fetched some blobs into that clone's object store (additive
  packs only; no history or working tree changed).
- The first study's artefacts were read but never modified.
