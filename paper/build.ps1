# Build the Isnad-AI IslamicEval-2026 Subtask-2 paper. LuaLaTeX required (Arabic via babel/Amiri).
param([string]$Name = "isnad_islamiceval2026_task2")
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
lualatex -interaction=nonstopmode "$Name.tex" | Out-Null
bibtex "$Name" | Out-Null
lualatex -interaction=nonstopmode "$Name.tex" | Out-Null
lualatex -interaction=nonstopmode "$Name.tex" | Out-Null
Write-Output "--- errors ---"
if (Test-Path "$Name.log") {
  $errs = Select-String -Path "$Name.log" -Pattern '^!' | ForEach-Object { $_.Line }
  if ($errs) { $errs } else { "none" }
  Write-Output "--- undefined refs/citations ---"
  Select-String -Path "$Name.log" -Pattern 'undefined' | ForEach-Object { $_.Line } | Select-Object -First 12
  Select-String -Path "$Name.log" -Pattern 'Output written' | ForEach-Object { $_.Line }
}
