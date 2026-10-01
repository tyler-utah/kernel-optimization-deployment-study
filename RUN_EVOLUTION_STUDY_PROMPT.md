# Mission

You are conducting an empirical software-engineering study for Tyler Sorensen's
talk, "Optimize the Optimization Pipeline." Work autonomously and persist until
the core deliverables below are complete. This is not a brainstorming task:
collect public data, write reproducible analysis code, validate the methodology,
run the analyses, and produce a careful preliminary report with figures.

The project root is:

`C:\Users\tsorensen\Documents\github\RealOptimizationTalk`

Read these files first:

1. `README.md`
2. `talk\talk-idea.md`
3. `experiments\README.md`
4. every file under `experiments\studies\`
5. `papers\related\related-work.md`

Do not edit the paper source material or unpublished-paper folders. Work only
under `experiments\`, except for adding a short link to a completed report from
the root README if appropriate.

# Central research question

How do open-source LLM inference systems actually evolve, and where does their
engineering effort go?

The motivating contrast is not merely "kernel work versus maintenance." Build a
taxonomy that exposes the optimization-delivery pipeline:

- kernel implementation and performance optimization
- kernel correctness fixes
- hardware and backend enablement
- model support
- core runtime, scheduling, caching, distributed execution, and serving features
- API, frontend, and router work
- tests, CI, build, benchmarking, and dependencies
- documentation
- refactoring, cleanup, and maintenance
- new workload families such as diffusion, multimodal generation, and RL

Treat "touches kernel code" as an orthogonal dimension, not as a mutually
exclusive purpose. A commit can touch a kernel because it optimizes it, fixes
it, ports it, integrates a new model, or refactors it. This two-dimensional view
is important.

# Questions to answer

## A. Evolution and allocation of effort

1. What fraction of commits and merged PRs belongs to each engineering category?
2. How has that mix changed quarterly since each project began?
3. What fraction touches kernel code, and what are those kernel-touching changes
   actually doing: optimization, correctness, portability, model integration,
   testing, or maintenance?
4. Is kernel/performance work growing, shrinking, or staying proportional as the
   systems mature?
5. Are bug-fix and maintenance shares growing with age?
6. What new subsystems appear over time (for example diffusion, routers,
   disaggregation, multimodal support), and how do they displace or add to
   earlier work?
7. How concentrated is work among contributors? Report contributor
   concentration overall and by category, but do not name or rank individual
   people in the public-facing report. Use aggregate statistics such as top-5
   share, Gini coefficient, and bus-factor-style counts.

## B. The optimization-delivery pipeline

8. How long do kernel/performance PRs take to receive a first review and merge,
   compared with other PRs?
9. Are they abandoned or reverted more often?
10. What evidence accompanies performance PRs: microbenchmarks, end-to-end
    throughput/latency, accuracy evaluation, numerical tests, multi-hardware
    results?
11. What do reviewers ask for: correctness, another GPU/backend, integration,
    benchmark methodology, maintainability, tests, documentation?
12. How much discovered optimization work is sitting in the open-PR queue?

## C. Rules, gatekeeping, and insider knowledge

13. What rules for an acceptable contribution are codified in CONTRIBUTING
    files, PR templates, CODEOWNERS, CI workflows, governance documents, labels,
    and bots?
14. Which rules are mechanically enforced, and which depend on reviewer
    judgment?
15. Who can trigger expensive GPU CI, approve kernel paths, and merge changes?
    Report role and ownership structure without turning the study into a ranking
    of individuals.
16. Are codified requirements actually followed? For measurable rules, calculate
    compliance rates. Examples:
    - required PR-title format
    - completed checklist sections
    - DCO / Signed-off-by trailers
    - benchmark evidence for `[Perf]` or `[Kernel]` PRs
    - tests accompanying source changes
    - CODEOWNERS approval where observable
    - required labels before full CI
17. What important acceptance criteria appear repeatedly in reviews but are not
    documented? These are candidates for "insider knowledge."

## D. Connection to papers and deployment (stretch, after A-C)

18. In a recent conference such as MLSys 2025, how many papers propose a
    kernel-style optimization?
19. Place those papers on the deployment-evidence ladder in
    `experiments\studies\04-paper-to-production.md` (L0-L5).
20. Conversely, for deployed kernel families in vLLM and SGLang, how many can be
    traced to papers, vendor libraries, company engineering, or community work?

# Initial scope

Core repositories:

- `vllm-project/vllm`
- `sgl-project/sglang`

Use these two for the rigorous core study. Add `ggml-org/llama.cpp` and
`flashinfer-ai/flashinfer` only as targeted comparisons after the core is
complete. Do not broaden to fourteen repositories before validating the method.

Use:

- full project history for quarterly evolution
- the most recent complete 12 months for headline composition and PR-lifecycle
  comparisons
- a 24-month window for reverts if the 12-month count is too small

Record exact cutoff dates and repository commit SHAs.

# Existing state: inspect before rerunning

Earlier work may already exist:

- blobless clones under `experiments\data\repos\vllm` and `...\sglang`
- commit logs under `experiments\data\`
- `experiments\scripts\effort_breakdown.py`
- CSVs and charts under `experiments\results\`

The earlier pilot analyzed roughly 22,106 vLLM and 19,040 SGLang first-parent
commits. Preliminary, UNVALIDATED last-12-month estimates were:

- vLLM: 10.7% kernel/perf purpose, 25.9% bug fixes, 12.8% hardware,
  17.0% touching kernel code
- SGLang: 8.1% kernel/perf purpose, 15.4% bug fixes, 12.4% hardware,
  11.5% touching kernel code

Do not quote these as results until validation. Inspect what exists, fix it
surgically, and reuse valid artifacts. Do not redo a slow operation merely
because it is convenient.

An earlier attempt to research PR rules in a background agent was cancelled and
may not have produced an artifact. Check
`experiments\studies\data-notes\pr-rules.md`; if absent or incomplete, do that
work yourself.

# Method requirements

## 1. Reproducibility and provenance

- Put scripts in `experiments\scripts\`.
- Put small, derived CSVs and figures in `experiments\results\`.
- Put methodological notes in `experiments\studies\data-notes\`.
- Put the main report at `experiments\system-evolution-report.md`.
- Add `experiments\METHODS.md` documenting data sources, API queries, cutoff
  dates, repository SHAs, exclusions, and known threats to validity.
- Cache GitHub API responses under `experiments\data\github\` so interrupted runs
  can resume without repeating requests.
- Keep large raw files out of the report. Do not commit cloned repositories or
  large API caches if this workspace later becomes a git repository.

## 2. Bounded, checkpointed execution

The previous interactive run became stuck during a cosmetic chart rerender.
Avoid opaque long-running calls:

- Use bounded commands and print progress.
- Put timeouts on network operations.
- Checkpoint after each phase by writing
  `experiments\results\study-status.json`.
- If a command will process thousands of records, make it resumable and write a
  progress counter.
- Do not start servers or watchers.
- Do not install new tools unless an existing dependency is missing and the
  installation is necessary.
- Do not recursively delete broad directories. Never delete user work.
- If a phase fails, record the failure and continue with independent work rather
  than losing the whole run.

### Checkpoint and resume contract

There are two independent checkpoint layers:

1. The Copilot CLI session is named `optimization-evolution-study` and can be
   resumed by the caller.
2. The study itself MUST persist enough state on disk that a completely new
   agent, with no conversation history, can resume correctly.

Create `experiments\results\study-status.json` before doing expensive work.
Update it atomically (write a temporary file, then replace the old file) after
every completed phase and after every batch of at most 250 GitHub records.

Use this schema:

```json
{
  "schema_version": 1,
  "updated_at_utc": "ISO-8601 timestamp",
  "study_cutoff_date": "YYYY-MM-DD",
  "repository_heads": {
    "vllm-project/vllm": "commit SHA",
    "sgl-project/sglang": "commit SHA"
  },
  "current_phase": "phase-id",
  "resume_next": "plain-language exact next action",
  "phases": {
    "00-audit-existing": {
      "status": "pending|in_progress|complete|blocked",
      "started_at_utc": null,
      "completed_at_utc": null,
      "outputs": [],
      "notes": ""
    }
  },
  "github_collection": {
    "repo": {
      "query_or_dataset": {
        "status": "pending|in_progress|complete|blocked",
        "cursor": null,
        "records_written": 0,
        "cache_files": []
      }
    }
  },
  "validation": {
    "iteration": 0,
    "sample_file": null,
    "labels_file": null,
    "metrics_file": null,
    "macro_f1": null,
    "kernel_perf_precision": null,
    "passed_threshold": false
  },
  "failures": []
}
```

The `phases` object must contain all phase IDs listed below. Every output path
must be relative to the project root. Do not mark a phase complete until each
listed output exists and has been sanity-checked.

At startup, always:

1. Read `study-status.json` if it exists.
2. Verify that outputs claimed as complete still exist.
3. Resume from `resume_next`; do not repeat completed network collection or
   validated analysis.
4. If the manifest and filesystem disagree, repair the manifest conservatively
   and record the discrepancy in `failures`.

### Required phases and concrete outputs

#### Phase 00 — Audit and freeze inputs

Outputs:

- `experiments\results\study-status.json`
- `experiments\results\repository-snapshot.json` containing repository URLs,
  HEAD SHAs, first/last commit dates, commit counts, and cutoff date
- `experiments\studies\data-notes\existing-artifacts-audit.md` explaining which
  previous files were reused, repaired, regenerated, or rejected

#### Phase 01 — Contribution rules and gatekeeping

Outputs:

- `experiments\studies\data-notes\pr-rules.md`
- `experiments\results\contribution-rules.csv`

The CSV has at least:

`repo,rule_id,rule,source,exact_quote,enforcement,measurable,measurement_signal`

#### Phase 02 — Taxonomy and measurement instrument

Outputs:

- `experiments\studies\data-notes\classification-codebook.md`
- repaired/reproducible classifier scripts under `experiments\scripts\`
- `experiments\results\classification-input-summary.json`

#### Phase 03 — Validation

Outputs:

- `experiments\results\validation-sample-vllm.csv`
- `experiments\results\validation-sample-sglang.csv`
- `experiments\results\validation-labels-vllm.csv`
- `experiments\results\validation-labels-sglang.csv`
- `experiments\results\classification-metrics.json`
- confusion-matrix figures for both repositories

If thresholds are not met, create versioned iteration files rather than
overwriting failed validation evidence.

#### Phase 04 — System evolution results

Outputs:

- `experiments\results\vllm-commits.csv`
- `experiments\results\sglang-commits.csv`
- `experiments\results\vllm-quarterly.csv`
- `experiments\results\sglang-quarterly.csv`
- `experiments\results\headline-numbers.csv`
- all required evolution figures under `experiments\results\figures\`

`headline-numbers.csv` must contain:

`finding_id,repo,metric,estimate,denominator,start_date,end_date,source_file,notes`

#### Phase 05 — PR lifecycle and review

Outputs:

- cached raw/batched responses under `experiments\data\github\`
- `experiments\results\pr-lifecycle-vllm.csv`
- `experiments\results\pr-lifecycle-sglang.csv`
- `experiments\results\pr-evidence-summary.csv`
- lifecycle figures under `experiments\results\figures\`

If API limits prevent complete collection, the manifest must contain the last
cursor and exact resume query. A documented, reproducible sample is acceptable
only after attempting the complete bounded collection.

#### Phase 06 — Report, methods, and consistency verification

Outputs:

- `experiments\METHODS.md`
- `experiments\system-evolution-report.md`
- `experiments\results\consistency-check.json`

`consistency-check.json` must list every numerical claim in the executive
summary and whether it exactly matches the cited generated CSV.

#### Phase 07 — Stretch: papers to deployment

This phase is optional until Phases 00–06 are complete.

Outputs, if run:

- `experiments\results\mlsys-2025-paper-census.csv`
- `experiments\results\deployment-evidence.csv`
- a section in the main report clearly labeled as a pilot

#### Final output index

Create `experiments\STUDY_OUTPUTS.md` last. It must be a short human-readable
index that says:

- what was completed and what remains
- which report to read first
- which five figures are most useful for the talk
- where the validated data and methods live
- the exact CLI resume command

## 3. Classification

Build a clear written codebook in
`experiments\studies\data-notes\classification-codebook.md`.

Use layered evidence:

1. contributor-chosen title tags and conventional-commit prefixes
2. changed paths and file types
3. subject/body keywords
4. LLM or manual interpretation for ambiguous records

Keep separate fields for:

- primary purpose
- touches kernel implementation
- touches kernel tests only
- hardware/backend
- optimization claim
- correctness-fix claim

Do not count a test, benchmark, or documentation path as kernel implementation
merely because its path contains "kernel."

## 4. Validation

Before presenting percentages:

- Draw a stratified random sample of at least 200 records per repository,
  oversampling kernel/perf, hardware, bug-fix, and ambiguous/path-classified
  records.
- Manually label the sample from the subject and changed paths using the
  codebook. Save the blinded sample and final labels.
- Report confusion matrices, per-class precision/recall/F1, macro F1, and
  bootstrap confidence intervals for headline shares.
- If macro F1 is below 0.80 or kernel/perf precision is below 0.90, refine the
  classifier and repeat validation. Preserve both iterations for transparency.
- For LLM-assisted judgments, preserve the prompt, model name, and raw output.
  Treat the classifier as a measurement instrument, not an oracle.

If true independent double-coding is unavailable, say so. Do not manufacture
inter-rater agreement.

## 5. Merge-history caveats

- vLLM and SGLang squash PRs into first-parent commits, and subjects usually
  contain PR numbers. Exploit that.
- PyTorch's merge bot makes GitHub's `is:merged` field unreliable, but PyTorch
  is not in the initial rigorous scope.
- Separate bot/dependency/release commits where appropriate.
- Account for partial quarters. Never show a partial current quarter as
  comparable to complete quarters without marking it.
- Normalize both counts and shares; rapid project growth can make either alone
  misleading.

## 6. PR and review data

Use authenticated `gh` and GitHub GraphQL/REST. Prefer batched GraphQL queries
and cached responses. Respect rate limits.

For PR lifecycle, build a dataset with at least:

- number, repo, title, author type (bot/human only; no public ranking)
- created, first non-author review/comment, closed, merged
- state and labels
- changed paths or kernel-touch flag
- review count and review rounds
- additions/deletions/files changed
- linked issue/revert/superseding PR where detectable
- benchmark/evaluation evidence fields
- category and classifier source/confidence

If complete review timelines are too costly, use a preregistered random sample
and explain the sampling scheme.

# Codified-rule audit

For each core repo, inspect at minimum:

- root and docs-site CONTRIBUTING material
- `.github\PULL_REQUEST_TEMPLATE*`
- `.github\CODEOWNERS`
- workflow files that gate PRs
- title checks, DCO/signoff checks, label-triggered CI, stale bots, auto-labelers
- governance, maintainer, reviewer, or committer documents
- any policy on AI-generated contributions
- any kernel/performance-specific benchmark or correctness requirements

Create a table:

`rule | source | exact quote | automatic/social | measurable signal | compliance result`

Also produce:

- number of owners/reviewer groups covering kernel paths
- whether expensive GPU CI requires maintainer action or a label
- who can merge by role/tier, without profiling individuals
- a short list of repeated review requirements that are not codified

# Analyses and figures

At minimum produce:

1. Quarterly stacked shares of engineering purpose for vLLM and SGLang, with
   absolute commit volume below. Use consistent colors across repos. Mark
   partial quarters.
2. Quarterly fraction of commits touching kernel implementation.
3. For kernel-touching commits, a breakdown of why they touched kernels:
   optimization, bug fix, hardware port, model integration, test/CI,
   refactor/maintenance, other.
4. A maturation view: bug-fix + maintenance share over project age.
5. Contributor concentration over time, aggregate only.
6. If PR data is completed: Kaplan-Meier or ECDF time-to-merge by category and
   open-PR age distribution.
7. A codified-rules versus observed-compliance table.

Every figure needs:

- clear denominator
- exact date range
- sample size
- validation quality or confidence interval where applicable
- a caption that distinguishes observation from interpretation

# Report structure

Write `experiments\system-evolution-report.md` with:

1. Executive summary: five defensible findings, each one sentence plus number
2. Research questions
3. Repositories and data
4. Taxonomy and validation
5. How the systems evolve
6. Where kernel work fits
7. PR delivery and review
8. Codified rules, gatekeeping, and undocumented expectations
9. Implications for Discover / Establish / Sustain
10. Threats to validity
11. Slide-ready findings
12. Next studies, including MLSys paper-to-deployment analysis

The report must clearly label preliminary results and must not turn correlations
into causal claims.

# Talk-oriented interpretation

Test, rather than assume, these possible claims:

- Kernel optimization is important but is a minority of total engineering work.
- Much kernel-touching work is not discovery; it is correctness, portability,
  integration, and maintenance.
- As serving systems mature, Establish and Sustain consume more work.
- The Optimization Gap is visible as queues, review requirements, reverts,
  hardware coverage, and CI infrastructure.
- Contributor concentration and gatekeeping may make deployment depend on scarce
  insider knowledge.

Be willing to report contrary evidence. If kernel optimization dominates, or
performance PRs merge faster than other PRs, that is an interesting result.

# Completion criteria

Do not stop at a plan. The core run is complete only when:

- existing artifacts have been inspected and repaired where needed
- the codebook and validated classifier exist
- vLLM and SGLang evolution results and figures exist
- the codified-rule audit exists
- at least a preliminary PR-lifecycle analysis exists, or a clearly documented
  rate-limit/data blocker exists with cached partial data
- `system-evolution-report.md` and `METHODS.md` exist
- scripts can rerun the analysis from cached/raw inputs
- a final consistency check confirms every headline number from the report
  matches a generated table

At the end, summarize:

- what was completed
- the strongest three findings
- classifier quality
- unresolved limitations
- exact files to open first
