param(
    [Parameter(Mandatory=$true)][string]$Config,
    [Parameter(Mandatory=$true)][string]$Dataset,
    [Parameter(Mandatory=$true)][string]$Root,
    [Parameter(Mandatory=$true)][ValidateSet('dev','train','test','robustness','real','smoke')][string]$Phase,
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
# Run from pt_reg. Explicit exit-code checks also work on Windows PowerShell 5.
$gaCommon = @('--config', $Config, '--dataset', $Dataset, '--root', $Root)
function Invoke-GA {
    param([string]$Command, [string[]]$Extra = @())
    & $Python -m generative_assembly $Command @gaCommon @Extra
    if ($LASTEXITCODE -ne 0) { throw "generative_assembly $Command failed with exit code $LASTEXITCODE" }
}
function Invoke-Phase {
    param([string]$Selected)
    switch ($Selected) {
        'dev' {
            Invoke-GA 'run' @('--split','dev','--stages','E0','E1','E2','E3','E4','E5')
            Invoke-GA 'evaluate' @('--split','dev')
        }
        'train' {
            Invoke-GA 'run' @('--split','train','--stages','E0','E1','E2','E4','E5')
            Invoke-GA 'train'
            Invoke-GA 'evaluate' @('--split','train')
            Invoke-GA 'run' @('--split','dev','--stages','E6')
            Invoke-GA 'evaluate' @('--split','dev')
        }
        'test' {
            Invoke-GA 'freeze'
            Invoke-GA 'run' @('--split','test','--stages','E0','E1','E2','E3','E4','E5','E6')
            Invoke-GA 'evaluate' @('--split','test')
            Invoke-GA 'verify'
        }
        'robustness' {
            Invoke-GA 'run' @('--split','test','--stages','E7')
            Invoke-GA 'evaluate' @('--split','test')
        }
        'real' {
            Invoke-GA 'run' @('--split','real','--stages','E0','E1','E2','E4','E5','E6','E7')
            Invoke-GA 'evaluate' @('--split','real')
        }
        'smoke' {
            foreach ($gaNextPhase in @('dev','train','test','robustness')) { Invoke-Phase $gaNextPhase }
        }
    }
}
Invoke-Phase $Phase
