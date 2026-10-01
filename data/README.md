# Data layout

## Versioned data

The repository versions derived tables, coding decisions, codebooks, figures,
and reports. These are the inputs required to regenerate published aggregate
numbers without network access or another agent run.

Important locations:

- `../results/`: preliminary-study tables and figures;
- `../deep-study/data/`: deep-study derived CSV/JSON datasets;
- `../deep-study/coding/`: adjudication batches and codebooks;
- `../kernel-lineage-study/data/`: assembled lineage datasets;
- `../kernel-lineage-study/staging/`: screening and coding outputs.

## Rebuildable ignored data

The following paths are intentionally ignored because they are large copies of
public sources or API responses:

- `repos/`: blobless clones of vLLM and SGLang at frozen commits;
- `github/`: resumable GitHub REST/GraphQL response pages;
- `../deep-study/data/cache/`: detailed API and intermediate caches;
- `../deep-study/data/src/`: frozen source checkouts used for audits;
- `../kernel-lineage-study/cache/`: Git, GitHub, blob, and release caches.

Recreate the core caches with:

```powershell
python scripts\fetch_data.py all
```

Then use `scripts/run_agents.ps1` for agent-coded stages that depend on those
caches. Every collector and coding workflow is resumable.

## Provenance

`../config/study.json` is the source of truth for repositories, date windows,
random seeds, and frozen commits. Study-specific `METHODS.md` and `STATUS.json`
files document sampling, coding, exclusions, and checkpoints.
