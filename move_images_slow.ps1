# Image move script (I/O throttled) - pure ASCII

# Find source: any folder under C:\temp_seg with Training that has jpg files
$srcCandidates = Get-ChildItem "C:\temp_seg" -Recurse -Directory | Where-Object { $_.FullName -like "*Training*01*" }
foreach ($c in $srcCandidates) {
    $jpgCount = (Get-ChildItem $c.FullName -Filter "*.jpg" -File -ErrorAction SilentlyContinue).Count
    if ($jpgCount -gt 1000) {
        $src = $c.FullName
        break
    }
}

# Find dest: any folder under D:\AIHub_dataset\075* with Training\01*
$dstCandidates = Get-ChildItem "D:\AIHub_dataset\075*" -Recurse -Directory -ErrorAction SilentlyContinue | Where-Object { $_.FullName -like "*Training*01*" }
foreach ($c in $dstCandidates) {
    if ($c.FullName -notlike "*02*") {
        $dst = $c.FullName
        break
    }
}

$batchSize = 500
$batchDelay = 2

Write-Host "=== Image Move ===" -ForegroundColor Cyan
Write-Host "Source: $src"
Write-Host "Dest: $dst"

if (-not $src) { Write-Host "ERROR: Source not found" -ForegroundColor Red; exit 1 }
if (-not $dst) { Write-Host "ERROR: Dest not found" -ForegroundColor Red; exit 1 }

$files = Get-ChildItem -Path $src -Filter "*.jpg" -File
$total = $files.Count
$moved = 0
$skipped = 0

Write-Host "Total: $total files"

foreach ($file in $files) {
    $dstPath = Join-Path $dst $file.Name
    if (Test-Path $dstPath) { $skipped++; continue }
    Move-Item -Path $file.FullName -Destination $dst -Force
    $moved++
    if ($moved % 1000 -eq 0) {
        $pct = [math]::Round(($moved + $skipped) / $total * 100, 1)
        Write-Host "[$pct%] Moved: $moved / Skipped: $skipped" -ForegroundColor Green
    }
    if ($moved % $batchSize -eq 0) { Start-Sleep -Seconds $batchDelay }
}

Write-Host "=== Done: Moved $moved, Skipped $skipped ===" -ForegroundColor Cyan
