# Methods: optimization-delivery evolution study

This document describes the reproducible preliminary study of
`vllm-project/vllm` and `sgl-project/sglang`. The analysis cutoff is
**2026-09-28**. Results are observational and should not be read as causal.

## Frozen inputs and provenance

| Repository | Frozen HEAD | First-parent dates | Cached first-parent records |
|---|---|---|---:|
| vLLM | `7230dfea501b232ce92d057f1c5ee9cb009c4cb4` | 2023-02-09–2026-09-29 | 22,106 before cutoff filtering |
| SGLang | `79cafec013d0a01c782d7a4bc902ef32a8941ba8` | 2023-10-09–2026-09-28 | 19,040 before cutoff filtering |

The exact repository URLs and snapshot checks are in
`results/repository-snapshot.json`. The vLLM snapshot contains ten commits
dated 2026-09-29 UTC; the analysis excludes them to preserve the stated
2026-09-28 cutoff. SGLang has five post-cutoff-excluded records under the same
date rule. The resulting populations are 22,096 and 19,035 first-parent
commits.

Name-only logs were produced with:

```powershell
git log --no-renames --first-parent `
  --format="@@@%H|%as|%ae|%s" --name-only HEAD
```

The raw logs are `data/vllm-log.txt` and `data/sglang-log.txt`. The cached local
Git object stores have intentionally empty working trees; no analysis assumes
that a checkout is present. The artifact audit is
`studies/data-notes/existing-artifacts-audit.md`.

## Time windows and units

- Full cutoff-filtered history is used for quarterly evolution.
- The headline window is the most recent complete 12 months,
  **2025-09-29 through 2026-09-28**, inclusive.
- 2026Q3 is marked partial because the cutoff precedes quarter end.
- A first-parent commit is the commit unit. Both projects generally squash
  merged PRs and preserve `(#number)` in the subject, so this is also a useful
  merged-PR proxy. It is not assumed to equal every merged PR.
- PR lifecycle uses PRs created in the same complete 12-month window. Open-queue
  age uses every PR still open at the cutoff, including older PRs.

## Classification instrument

`scripts/system_evolution.py` implements version 4.0 of the codebook in
`studies/data-notes/classification-codebook.md`. It uses layered evidence:
contributor title tags/conventional prefixes, changed production paths, title
keywords, and a conservative fallback. The output separates:

- one primary engineering purpose;
- kernel implementation touch;
- kernel-tests-only touch;
- hardware/backend evidence;
- optimization claim;
- correctness-fix claim; and
- the purpose of a kernel implementation touch.

Test, benchmark, documentation, examples, CI, and tooling paths are excluded
from kernel implementation merely containing the word `kernel`. Public commit
outputs replace author emails with repository-scoped 16-character SHA-256
prefixes. Aggregate contributor results therefore do not name or rank people.

## Validation

The script uses a fixed seed (`20260928 + iteration - 1`) to draw 220 records
per repository per iteration.
Sampling oversamples kernel touches, kernel performance, kernel correctness,
hardware/backend work, bug fixes, and low-confidence/path classifications. The
blinded files contain subjects and changed paths but omit predictions. A single
coder labels primary purpose and kernel implementation touch using the written
codebook. Independent double-coding was unavailable, so this study does **not**
report inter-rater agreement.

`scripts/system_evolution.py metrics` produces confusion matrices, per-class
precision/recall/F1, macro F1, kernel-performance precision, and kernel-touch
precision/recall. The preregistered acceptance thresholds are macro F1 at least
0.80 and kernel-performance precision at least 0.90 in each repository.
Four fresh-sample iterations were run. Each failed the macro-F1 release gate;
the `-v1` through `-v4` samples, labels, metrics, and confusion matrices are
retained rather than silently overwritten. The final v4 holdout reached macro
F1 0.677 for vLLM and 0.668 for SGLang. Kernel-performance precision did pass
its separate gate (0.955 and 1.000; recall 0.875 and 0.778), so the report
promotes only that purpose-specific estimate. Fine-purpose allocations and the
kernel-touch-reason decomposition remain diagnostic. Kernel-touch
precision/recall was 0.885/0.708 and 0.667/0.618, respectively.

Headline share uncertainty is a nonparametric 2,000-replicate record bootstrap
with the same fixed seed. These intervals quantify sampling variation in the
observed commit population; they do not incorporate classifier error,
repository selection, or missing-PR uncertainty.

## Contributor concentration

Per-quarter and recent-window statistics include:

- top-five contributor share;
- Gini coefficient over commit counts; and
- the smallest number of contributors accounting for at least 50% of commits.

The same statistics are computed within engineering categories. Distinct public
author emails are not identity-resolved, so aliases can bias concentration
downward. Bot/dependency/release commits remain visible in primary outputs and
are discussed separately where material.

## GitHub PR collection

Collection requires the authenticated `gh` CLI. The resumable collector is
`scripts/pr_lifecycle.py`.

1. It pages the REST endpoint
   `GET /repos/{owner}/{repo}/pulls?state=all&sort=created&direction=desc`
   in 100-record batches until it crosses the start date.
2. It separately pages `state=open` to enumerate the complete open queue at the
   cutoff.
3. It preregisters a deterministic 600-PR review-timeline sample per repository,
   stratified by state and oversampling kernel performance, kernel correctness,
   hardware/backend, and bug-fix purposes.
4. In 20-PR GraphQL batches it requests PR metadata, first 100 reviews, first
   100 issue comments, labels, additions, deletions, and changed-file count.

Every response is cached under `data/github/{repo}/`; the checkpoint manifest
is atomically replaced after every batch of at most 100 REST or 20 detailed
records. Network calls have a 75-second timeout and three bounded attempts.
Reruns reuse existing cache files.

The first review time is the first review or issue comment by a non-bot account
other than the PR author. Review rounds count distinct non-author,
non-bot reviewer-days.
Closed-unmerged PRs and open PRs older than 90 days are labeled abandoned for
the preliminary comparison; superseding-link signals are retained because some
closed PRs landed through another PR.

Evidence and reviewer-concern fields are transparent regular-expression text
extractors over the PR body and non-bot reviews/comments. They detect
microbenchmark, end-to-end, accuracy, numerical-test, and multi-hardware
evidence, plus correctness, hardware, integration, benchmark-method,
maintainability, test, and documentation concerns. These are auditable lower
bounds, not semantic judgments. Detailed timelines with more than 100 reviews
or comments are flagged as truncated.

Lifecycle and evidence shares are post-stratified by the inverse observed
category-by-state sampling fraction to recover recent-PR population estimates.
Within-state time-to-merge ECDFs are unaffected by this sampling allocation.

The full recent PR metadata collection was attempted. Detailed review timelines
for every PR would exceed a reasonable bounded API budget, so lifecycle and
review-text comparisons use the preregistered sample; open-queue counts use the
complete REST listing.

## Contribution-rule audit and compliance

The audit freezes contribution documents, PR templates, CODEOWNERS, governance,
and CI workflows at the repository SHAs above. `results/contribution-rules.csv`
records the source, exact quote, enforcement mechanism, measurability, and
observable signal. The narrative distinguishes mechanically enforced checks
from reviewer judgment and reports roles/groups rather than ranking people.

`results/rule-compliance-observed.csv` reports only signals observable in the
lifecycle sample: leading title tag, checklist completion, benchmark evidence
for performance PRs, test/benchmark paths accompanying matched source commits,
vLLM squash-commit `Signed-off-by` trailers, and a non-author approval. Rates
are post-stratified to the recent-PR population. Squash trailers can undercount
DCO compliance on pre-squash commits. A current label cannot prove that the
label existed before expensive CI, and ordinary approval cannot prove CODEOWNERS status.
Those measurements are explicitly unavailable rather than inferred.

## Analysis and rerun

From the project root:

```powershell
python experiments\scripts\system_evolution.py classify
python experiments\scripts\system_evolution.py sample
# Fill the two validation-labels CSVs using the codebook.
python experiments\scripts\system_evolution.py metrics
python experiments\scripts\system_evolution.py analyze
python experiments\scripts\system_evolution.py figures
python experiments\scripts\pr_lifecycle.py all
```

`results/study-status.json` is the cross-session source of truth. It records
frozen SHAs, phase outputs, cache cursors, failures, and the exact next action.

## Exclusions and threats to validity

1. **Repository selection.** Two rapidly changing Python/CUDA inference engines
   do not represent all optimization systems, vendors, or project governance.
2. **Squash-history view.** First-parent history suppresses intermediate
   commits and may omit merged work delivered through synchronization or
   nonstandard merge paths.
3. **Purpose is latent.** One primary purpose simplifies multi-purpose changes.
   Orthogonal fields mitigate but cannot eliminate this measurement error.
4. **Paths evolve.** Kernel and subsystem layouts change. Static path rules can
   miss generated kernels or overcount dispatch/integration code.
5. **Single coding.** Validation is manually coded once, not independently
   double-coded. The classifier and coder use the same written construct.
6. **PR sampling.** Lifecycle results are from a purpose/state-stratified sample,
   not raw population proportions. Open-queue counts are complete, but title-only
   classification is conservative and cannot observe kernel paths.
7. **Review observability.** Private discussion, external CI, deleted comments,
   team membership, merge-queue policy, and label timing are not fully public.
8. **Text extraction.** Absence of benchmark/evaluation language is not proof
   that the author or reviewer performed no evaluation.
9. **Contributor identity.** Unresolved email aliases and bots affect aggregate
   concentration. No inference about individual productivity is warranted.
10. **Censoring and causality.** Open PRs are right-censored; category
    differences can reflect complexity, authorship, size, or governance. This
    preliminary analysis does not attribute causes.
11. **Reverts.** Public titles and links do not reliably identify the original
    reverted PR. Revert/rollback text is reported as a signal, not a validated
    category-specific revert rate.
