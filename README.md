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

The study asks a simple question: how does performance work move through these
repositories compared with everything else?

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

## Some results

### Performance PRs move more slowly

Among merged PRs, median time to merge was:

| Repository | Performance | Other |
|---|---:|---:|
| vLLM | 5.2 days | 1.5 days |
| SGLang | 4.0 days | 0.7 days |

Using cumulative incidence with closure as a competing event, performance PRs
were less likely to merge within 30 days:

- vLLM: **8.5 percentage points lower**
- SGLang: **10.1 percentage points lower**

### Performance PRs are larger

Median lines changed:

| Repository | Performance | Other |
|---|---:|---:|
| vLLM | 223 | 62 |
| SGLang | 274 | 78 |

### Performance work accumulates in the open queue

At the cutoff, the complete open queue contained **10,291 PRs**, including
**1,806 confirmed performance PRs**.

### Performance PRs are reverted more often

| Repository | Performance | Other |
|---|---:|---:|
| vLLM | 1.8% | 0.9% |
| SGLang | 2.4% | 1.1% |

### PR descriptions emphasize speed more than durability

Across the 6,308 confirmed performance PRs:

- 83.2% explicitly mentioned performance;
- 58.3% mentioned testing;
- 12.6% mentioned correctness without also mentioning testing; and
- 1.3% explicitly mentioned maintainability or forward/backward compatibility.

These are literal language matches after removing Markdown headings, not
semantic judgments or evidence that an unmentioned concern was ignored.

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
