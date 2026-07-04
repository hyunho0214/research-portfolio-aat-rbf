param()

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot

try {
    Write-Host "== Compile Python runner =="
    py -3 -m py_compile scripts\run_sql_practice.py

    Write-Host "== Execute SQLite SQLD practice queries =="
    py -3 scripts\run_sql_practice.py

    $requiredOutputs = @(
        "outputs\00_table_overview.csv",
        "outputs\01_defect_wafer_filter.csv",
        "outputs\02_pm_yield_relation.csv",
        "outputs\03_wafer_map_defect_preprocess.csv",
        "outputs\04_equipment_low_yield_summary.csv",
        "outputs\run_summary.md"
    )

    foreach ($path in $requiredOutputs) {
        if (-not (Test-Path $path)) {
            throw "Missing expected output: $path"
        }
    }

    Write-Host "Validation completed."
}
finally {
    Pop-Location
}
