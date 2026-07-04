param(
    [ValidateSet("compile", "smoke", "benchmark")]
    [string]$Mode = "compile"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot

try {
    Write-Host "== Compile Python entry points =="
    py -3 -m py_compile `
        run_experiment.py `
        tune_model.py `
        threshold_analysis.py `
        scripts\compare_reports.py `
        src\secom_defect\data.py `
        src\secom_defect\modeling.py `
        src\secom_defect\reporting.py

    if ($Mode -eq "smoke" -or $Mode -eq "benchmark") {
        Write-Host "== Smoke benchmark: no SMOTE, median imputer, top 30 features =="
        py -3 run_experiment.py --download --output-dir reports\validation_smoke --splits 3 --top-k 30 --imputer median --resampling none
    }

    if ($Mode -eq "benchmark") {
        Write-Host "== SMOTE benchmark: median imputer, top 30 features =="
        py -3 run_experiment.py --download --output-dir reports\validation_smote_median_top30 --splits 3 --top-k 30 --imputer median --resampling smote

        Write-Host "== SMOTE benchmark: KNN imputer, top 30 features =="
        py -3 run_experiment.py --download --output-dir reports\validation_smote_knn_top30 --splits 3 --top-k 30 --imputer knn --resampling smote

        Write-Host "== Compare validation reports =="
        py -3 scripts\compare_reports.py `
            --reports reports\validation_smoke reports\validation_smote_median_top30 reports\validation_smote_knn_top30 `
            --names no_smote_median smote_median smote_knn `
            --output-dir reports\validation_comparison
    }

    Write-Host "Validation completed: $Mode"
}
finally {
    Pop-Location
}
