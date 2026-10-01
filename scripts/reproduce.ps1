[CmdletBinding()]
param(
    [ValidateSet("check", "collect", "agents", "analysis", "all")]
    [string]$Stage = "check",
    [ValidateSet("preliminary", "deep", "lineage", "all")]
    [string]$Study = "all"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    python scripts/artifact_config.py
    if ($LASTEXITCODE -ne 0) { throw "Configuration validation failed." }
    python scripts/capture_environment.py
    if ($LASTEXITCODE -ne 0) { throw "Environment capture failed." }

    if ($Stage -eq "check") {
        python scripts/talk_numbers.py
        if ($LASTEXITCODE -ne 0) { throw "Talk-number reproduction failed." }
        python scripts/verify_consistency.py
        if ($LASTEXITCODE -ne 0) { throw "Preliminary consistency checks failed." }
        python deep-study/scripts/consistency.py
        if ($LASTEXITCODE -ne 0) { throw "Deep-study consistency checks failed." }
        python kernel-lineage-study/scripts/consistency.py
        if ($LASTEXITCODE -ne 0) { throw "Lineage consistency checks failed." }
    }

    if ($Stage -in @("collect", "all")) {
        python scripts/fetch_data.py all
        if ($LASTEXITCODE -ne 0) { throw "Data collection failed." }
    }

    if ($Stage -in @("agents", "all")) {
        $agentStudy = switch ($Study) {
            "preliminary" { "evolution" }
            "deep" { "deep-study" }
            default { $Study }
        }
        & "$PSScriptRoot\run_agents.ps1" -Study $agentStudy
    }

    if ($Stage -in @("analysis", "all")) {
        python scripts/run_analysis.py $Study
        if ($LASTEXITCODE -ne 0) { throw "Analysis failed." }
    }
}
finally {
    Pop-Location
}
