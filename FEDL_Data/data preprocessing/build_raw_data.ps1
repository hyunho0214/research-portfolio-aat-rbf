param(
    [string]$InputDir = $PSScriptRoot,
    [string]$OutputCsv = 'raw_data.csv',
    [string[]]$SelectedColumns = @('source_file', 'source_line', 'setup_title', 'V1', 'Abs_Id')
)

$ErrorActionPreference = 'Stop'

$outPath = Join-Path $InputDir $OutputCsv

$csvFiles = Get-ChildItem -Path $InputDir -Filter '*.csv' |
    Where-Object { $_.Name -ne $OutputCsv } |
    Sort-Object Name

if (-not $csvFiles) {
    throw 'No input CSV files were found.'
}

$allBlocks = New-Object System.Collections.Generic.List[object]

foreach ($file in $csvFiles) {
    $lines = Get-Content -Path $file.FullName

    $currentTitle = ''
    $currentTitleLine = 0

    for ($i = 0; $i -lt $lines.Count; $i++) {
        $line = $lines[$i]
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }

        if ($line -like 'SetupTitle,*') {
            $parts = $line.Split(',')
            if ($parts.Count -ge 2) {
                $currentTitle = $parts[1].Trim()
                $currentTitleLine = $i + 1
            }
            continue
        }

        if ($line -like 'AnalysisSetup, Analysis.Setup.Title,*') {
            $parts = $line.Split(',')
            if ($parts.Count -ge 3) {
                $currentTitle = $parts[2].Trim()
                $currentTitleLine = $i + 1
            }
            continue
        }

        if ($line -like 'DataName,*') {
            $parts = $line.Split(',')
            $columns = @()
            for ($k = 1; $k -lt $parts.Count; $k++) {
                $columns += $parts[$k].Trim()
            }

            $blockRows = New-Object System.Collections.Generic.List[object]
            $j = $i + 1
            while ($j -lt $lines.Count) {
                $dataLine = $lines[$j]
                if (-not ($dataLine -like 'DataValue,*')) {
                    break
                }

                $dparts = $dataLine.Split(',')
                $row = [ordered]@{
                    source_file = $file.Name
                    source_line = $j + 1
                    setup_title = $currentTitle
                }

                for ($k = 0; $k -lt $columns.Count; $k++) {
                    $idx = $k + 1
                    $val = ''
                    if ($idx -lt $dparts.Count) {
                        $val = $dparts[$idx].Trim()
                    }
                    $row[$columns[$k]] = $val
                }

                $blockRows.Add([pscustomobject]$row) | Out-Null
                $j++
            }

            $allBlocks.Add([pscustomobject]@{
                source_file = $file.Name
                setup_title = $currentTitle
                setup_title_line = $currentTitleLine
                data_name_line = $i + 1
                rows = $blockRows
            }) | Out-Null

            $i = $j - 1
            continue
        }
    }
}

# File keeps appending downward, so lower setup_title appears earlier in time.
$blocksOrdered = $allBlocks |
    Sort-Object @{Expression = 'source_file'; Ascending = $true}, @{Expression = 'setup_title_line'; Ascending = $false}, @{Expression = 'data_name_line'; Ascending = $false}

$outputRows = New-Object System.Collections.Generic.List[object]
$index = 0

for ($b = 0; $b -lt $blocksOrdered.Count; $b++) {
    $block = $blocksOrdered[$b]
    $rowsInBlock = $block.rows | Sort-Object @{Expression = 'source_line'; Ascending = $true}

    foreach ($r in $rowsInBlock) {
        $index++
        $obj = [ordered]@{ index = $index }
        foreach ($col in $SelectedColumns) {
            $obj[$col] = ''
            if ($r.PSObject.Properties.Name -contains $col) {
                $obj[$col] = $r.$col
            }
        }
        $outputRows.Add([pscustomobject]$obj) | Out-Null
    }

    # Insert a blank index row between setup_title blocks.
    if ($b -lt $blocksOrdered.Count - 1) {
        $gap = [ordered]@{ index = '' }
        foreach ($col in $SelectedColumns) {
            $gap[$col] = ''
        }
        $outputRows.Add([pscustomobject]$gap) | Out-Null
    }
}

$outputRows | Export-Csv -Path $outPath -NoTypeInformation -Encoding UTF8

Write-Output ('Created: ' + $outPath)
Write-Output ('Rows: ' + $outputRows.Count)
Write-Output ('Blocks: ' + $blocksOrdered.Count)
Write-Output ('Input files: ' + $csvFiles.Count)
Write-Output ('Columns: index, ' + ($SelectedColumns -join ', '))
