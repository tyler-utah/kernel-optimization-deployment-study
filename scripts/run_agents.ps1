[CmdletBinding()]
param(
    [ValidateSet("deep-study", "lineage", "evolution", "all")]
    [string]$Study = "all",
    [string]$Model = "auto",
    [ValidateSet("low", "medium", "high", "xhigh", "max")]
    [string]$ReasoningEffort = "high",
    [int]$MaxAutopilotContinues = 20,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$legacyRoot = "C:\Users\tsorensen\Documents\github\RealOptimizationTalk"
$runDir = Join-Path $root "agent-runs"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$prompts = @{
    "evolution" = "RUN_EVOLUTION_STUDY_PROMPT.md"
    "deep-study" = "RUN_DEEP_STUDIES_PROMPT.md"
    "lineage" = "RUN_KERNEL_LINEAGE_STUDY_PROMPT.md"
}

if (-not $DryRun -and -not (Get-Command copilot -ErrorAction SilentlyContinue)) {
    throw "GitHub Copilot CLI is required. Install it and run 'copilot login'."
}

$selected = if ($Study -eq "all") {
    @("evolution", "deep-study", "lineage")
} else {
    @($Study)
}

foreach ($name in $selected) {
    $promptPath = Join-Path $root $prompts[$name]
    $prompt = Get-Content -LiteralPath $promptPath -Raw

    # Make legacy prompts portable when experiments is cloned independently.
    $prompt = $prompt.Replace("experiments\", "")
    $prompt = $prompt.Replace("experiments/", "")
    $prompt = $prompt.Replace($legacyRoot, $root)
    $prompt = @"
You are reproducing the $name study in a standalone artifact repository.
The artifact root is:
$root

Read config\study.json first. Its repositories, dates, seeds, and frozen SHAs
override any duplicated legacy constants in the prompt. Reuse existing
checkpoints and never overwrite human- or agent-coded outputs with guesses.

$prompt
"@

    if ($DryRun) {
        Write-Output "$name prompt ready ($($prompt.Length) characters)"
        continue
    }

    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $sharePath = Join-Path $runDir "$name-$stamp.md"
    & copilot `
        -C $root `
        --prompt $prompt `
        --model $Model `
        --reasoning-effort $ReasoningEffort `
        --autopilot `
        --max-autopilot-continues $MaxAutopilotContinues `
        --allow-all-tools `
        --allow-all-paths `
        --allow-all-urls `
        --share $sharePath
    if ($LASTEXITCODE -ne 0) {
        throw "Copilot agent failed for $name with exit code $LASTEXITCODE."
    }
}
