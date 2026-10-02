# Kernel Optimization Deployment Study

> **This study and all of its code were generated entirely by AI agents under
> human direction.**

This repository studies performance-optimization pull requests in two
high-throughput open-source inference systems:

- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)

It contains the coded data, analysis scripts, and reproducibility machinery for
a complete one-year population of pull requests created from September 29,
2025 through September 28, 2026.

## What we studied

We collected every pull request created during the study window:

| Repository | Created | Merged | Closed without merge | Still open |
|---|---:|---:|---:|---:|
| vLLM | 25,573 | 11,980 | 7,804 | 5,789 |
| SGLang | 26,004 | 13,871 | 7,696 | 4,437 |
| **Combined** | **51,577** | **25,851** | **15,500** | **10,226** |

That is approximately 70 PRs created and 33 merged per day in vLLM,
71 created and 38 merged per day in SGLang, and 42 combined PRs closed without
merge per day.

The study asks a simple question: how does performance work move through these
repositories compared with everything else?

### How the population was collected

The collector asks the GitHub API for every vLLM and SGLang pull request
created in the one-year window. It records whether each PR had merged, closed
without merging, or remained open at the fixed cutoff. API pages are cached so
an interrupted collection can resume without starting over.

The population counts are not a sample. They cover all PRs returned for the
two repositories and date window. The frozen dates, repositories, and commit
SHAs are in [`config/study.json`](config/study.json).

## How we partitioned pull requests

Each pull request is classified as either:

- **Performance:** its stated purpose is to improve latency, throughput,
  memory use, utilization, kernel efficiency, or another performance metric.
- **Non-performance:** all other changes, including features, bug fixes,
  documentation, tests, CI, refactoring, and maintenance without a performance
  objective.

The classifier first applies transparent repository-specific signals:

- explicit tags such as `[Perf]`, `[Performance]`, or `perf:`;
- performance labels;
- optimization language in the title or description; and
- changes to production kernel and performance-sensitive paths.

High-confidence tagged changes are accepted directly. Ambiguous candidates are
read in context by coding agents using a versioned codebook. Seeded samples of
both accepted and excluded records are also coded to estimate misses and false
positives. Tests, benchmarks, documentation, examples, CI, and tooling are not
treated as kernel implementations merely because their paths contain the word
`kernel`.

The final partition contains **6,308 confirmed performance PRs**:

- vLLM: 3,214
- SGLang: 3,094

See:

- [`deep-study/coding/codebook-a1-performance.md`](deep-study/coding/codebook-a1-performance.md)
- [`deep-study/data/performance-pr-population.csv`](deep-study/data/performance-pr-population.csv)
- [`deep-study/data/a1-tier-validation.csv`](deep-study/data/a1-tier-validation.csv)

## Results and methodology

One deterministic command reads the versioned study tables and reproduces
every headline result below:

```powershell
python scripts\headline_results.py
```

It writes:

- [`results/headline-results.md`](results/headline-results.md), for people; and
- [`results/headline-results.json`](results/headline-results.json), for tools.

The following sections explain where each result comes from and what it means.

### Performance PRs move more slowly

**Method.** For descriptive medians, we compare time from creation to merge
among PRs that eventually merged. Because that comparison leaves open PRs
unresolved, the main analysis also estimates the probability of merging within
30 days. It treats closure without merge as a competing outcome rather than
pretending that closed PRs could still merge. This uses every PR in the
one-year population and does not infer why an individual PR moved slowly.

The cohort contains **2,700 merged performance PRs**, or **10.4% of all merged
PRs**.

Among merged PRs, median time to merge was:

| Repository | Performance | Other |
|---|---:|---:|
| vLLM | 5.2 days | 1.5 days |
| SGLang | 4.0 days | 0.7 days |

Using cumulative incidence with closure as a competing event, performance PRs
were less likely to merge within 30 days:

- vLLM: **8.5 percentage points lower**
- SGLang: **10.1 percentage points lower**

Closed-without-merge shares were similar:

- vLLM: 28.5% for performance versus 30.7% for other PRs;
- SGLang: 28.7% versus 29.7%.

The larger difference was in PRs still open at the cutoff:

- vLLM: 31.8% for performance versus 21.3% for other PRs;
- SGLang: 25.3% versus 16.0%.

Sources:
[`a2-population-summary.csv`](deep-study/data/a2-population-summary.csv) and
[`a2-survival.csv`](deep-study/data/a2-survival.csv).

### Performance PRs are larger

**Method.** GitHub reports additions, deletions, and files changed for every PR,
so the line and file medians use the complete population. Commit counts and
review-message counts come from the detailed review corpus: up to 1,000
performance PRs per repository plus a non-performance comparison group matched
on creation month, size, and outcome. Reviewer messages exclude bots, the PR
author, approvals without discussion, and CI-command comments. The review
comparison is restricted to PRs with at least one substantive human message.

Median lines changed:

| Repository | Performance | Other |
|---|---:|---:|
| vLLM | 223 | 62 |
| SGLang | 274 | 78 |

The corresponding medians were:

| Repository | Files | Commits | Substantive reviewer messages |
|---|---:|---:|---:|
| vLLM | 4 vs 2 | 3 vs 2 | 3 vs 2 |
| SGLang | 4 vs 2 | 4 vs 3 | 3 vs 2 |

Each cell reports performance versus comparison PRs.

Sources:
[`a2-population-summary.csv`](deep-study/data/a2-population-summary.csv),
[`a3-evidence-summary.csv`](deep-study/data/a3-evidence-summary.csv), and
[`a3-review-message-summary.csv`](deep-study/data/a3-review-message-summary.csv).

### Performance work accumulates in the open queue

**Method.** The collector separately enumerates every PR still open at the
cutoff, including PRs created before the one-year cohort. The same partitioning
rules are applied to identify confirmed performance work. This is why the
complete queue count differs from the number of one-year-cohort PRs that were
still open.

At the cutoff, the complete open queue contained **10,291 PRs**, including
**1,806 confirmed performance PRs (17.5%)** and **8,485 other PRs**.

Source:
[`a1-population-summary.csv`](deep-study/data/a1-population-summary.csv).

### Performance PRs are reverted more often

**Method.** A separate two-year first-parent commit census finds explicit
reverts and rollbacks, links them back to the reverted PR, and codes both the
kind of reverted change and the stated reason. The one-year rates below use
merged PRs from the main cohort as the denominator. The reason breakdown uses
all 102 confirmed performance-change reverts found in the two-year census.

| Repository | Performance | Other |
|---|---:|---:|
| vLLM | 1.8% | 0.9% |
| SGLang | 2.4% | 1.1% |

Only **5 of 102** performance-change reverts cited a performance regression.
More common stated reasons included correctness or accuracy, CI or test
failures, hardware-specific breakage, crashes or hangs, and build or dependency
problems. Some reverts gave no reason.

Sources:
[`a2-revert-rates.csv`](deep-study/data/a2-revert-rates.csv) and
[`b1-revert-summary.csv`](deep-study/data/b1-revert-summary.csv).

### PR descriptions emphasize speed more than durability

**Method.** This is a deliberately narrow lexical analysis of all 6,308
confirmed performance PR titles and descriptions. Before matching, the script
removes Markdown headings so prefilled template labels such as “Test Plan” do
not count by themselves. The regular expressions then look for explicit
performance, testing, correctness, maintenance, and compatibility language.

Across the 6,308 confirmed performance PRs:

- 83.2% explicitly mentioned performance;
- 58.3% mentioned testing;
- 12.6% mentioned correctness without also mentioning testing; and
- 1.3% explicitly mentioned maintainability or forward/backward compatibility.

These are literal language matches after removing Markdown headings, not
semantic judgments or evidence that an unmentioned concern was ignored.

Source:
[`a6-pr-description-mentions.csv`](deep-study/data/a6-pr-description-mentions.csv).

The analyses are observational. They do not establish that performance work
causes longer reviews, larger changes, or reverts.

## Verify the checked-in results

Requirements:

- Python 3.11 or newer
- Git

On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\scripts\reproduce.ps1 -Stage check
```

The verification regenerates the headline tables and checks every
machine-readable claim against the checked-in derived data.

Important outputs:

- [`results/`](results/)
- [`deep-study/data/`](deep-study/data/)
- [`deep-study/SYNTHESIS.md`](deep-study/SYNTHESIS.md)
- [`kernel-lineage-study/SYNTHESIS.md`](kernel-lineage-study/SYNTHESIS.md)

## Run the study again

Full collection additionally requires:

- authenticated GitHub CLI (`gh auth login`);
- GitHub Copilot CLI for the agent-coded stages; and
- enough time and disk space for repository clones and resumable API caches.

```powershell
# Download the frozen repository and GitHub inputs.
.\scripts\reproduce.ps1 -Stage collect

# Re-run contextual coding and adjudication.
.\scripts\reproduce.ps1 -Stage agents

# Regenerate tables, figures, reports, and consistency checks.
.\scripts\reproduce.ps1 -Stage analysis
```

Large clones, API responses, and source snapshots are intentionally ignored by
Git. The collectors recreate them from [`config/study.json`](config/study.json)
and resume from existing cache pages.

## Extend the study

To use another date window or newer repository snapshots:

1. Update the dates, seeds, and frozen commit SHAs in
   [`config/study.json`](config/study.json).
2. Run collection, agent coding, and analysis in that order.
3. Keep the generated consistency reports with the resulting data.

Adding another repository also requires defining its repository-specific title,
label, and path rules in the population classifier. The existing vLLM and
SGLang rules provide the templates.

Detailed sampling, coding, survival-analysis, and validation procedures are in
[`deep-study/METHODS.md`](deep-study/METHODS.md).
