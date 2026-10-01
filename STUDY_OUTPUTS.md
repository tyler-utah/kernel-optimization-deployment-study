# Study outputs

Read [`system-evolution-report.md`](system-evolution-report.md) first, then
[`METHODS.md`](METHODS.md) for provenance, sampling, and threats to validity.

## Completed and remaining

Phases 00–02 and 04–06 are complete: frozen repository snapshots, contribution
rules, a two-dimensional taxonomy, full-history commit tables, quarterly
evolution outputs, contributor concentration, complete recent/open PR metadata,
a 600-PR detailed lifecycle sample per repository, rule-compliance signals,
figures, methods, report, and an exact executive-claim consistency check.

Validation ran four fresh 220-record-per-repository iterations. The final
kernel-performance class passed its precision gate in both repositories, but
the fine-purpose classifier did not pass macro F1 (0.677 vLLM, 0.668 SGLang
versus the 0.80 gate). Fine-purpose and kernel-touch-reason outputs are
therefore diagnostic, not talk-ready. Independent human recoding and classifier
refinement remain. The optional MLSys 2025 paper-to-production pilot was not
run.

## Five useful figures

1. [`pr-time-to-merge-ecdf.png`](results/figures/pr-time-to-merge-ecdf.png) — validated performance-purpose delivery comparison.
2. [`open-pr-age-distribution.png`](results/figures/open-pr-age-distribution.png) — complete open-queue age.
3. [`codified-rules-compliance.png`](results/figures/codified-rules-compliance.png) — automatic rules versus observed signals.
4. [`contributor-concentration.png`](results/figures/contributor-concentration.png) — aggregate concentration without naming people.
5. [`quarterly-purpose-vllm.png`](results/figures/quarterly-purpose-vllm.png) — useful method/diagnostic view, but **not slide-ready** until fine-purpose validation passes.

## Data and methods

- Validated performance-specific metrics and all diagnostic headline values:
  [`results/headline-numbers.csv`](results/headline-numbers.csv)
- Final and versioned validation evidence:
  [`results/classification-metrics.json`](results/classification-metrics.json)
  and `results/validation-*-v1` through `-v4`
- Commit populations and quarterly tables: `results/vllm-*.csv` and
  `results/sglang-*.csv`
- PR lifecycle and evidence: `results/pr-lifecycle-*.csv` and
  [`results/pr-evidence-summary.csv`](results/pr-evidence-summary.csv)
- Rules and compliance:
  [`results/contribution-rules.csv`](results/contribution-rules.csv) and
  [`results/rule-compliance-observed.csv`](results/rule-compliance-observed.csv)
- Reproducible scripts: [`scripts/`](scripts/)
- Persistent checkpoint:
  [`results/study-status.json`](results/study-status.json)

## Resume

```powershell
copilot --resume="optimization-evolution-study"
```

The exact next study action is also stored in
`results/study-status.json`: independently recode a fresh validation sample,
then refine only the confused model/core/API/maintenance boundaries before
promoting fine-purpose shares.
