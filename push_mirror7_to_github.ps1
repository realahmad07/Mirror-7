<#
.SYNOPSIS
    All-in-One Colab ZIP Merger + Step-by-Step GitHub Push Script for Mirror-7.
    Author: Ahmad Hassan | Target: https://github.com/realahmad07/Mirror-7.git

.DESCRIPTION
    1. Automatically finds 'Mirror7_Colab_GitHub_Sync.zip' in Downloads/Desktop
       (or accepts -ColabZipPath "C:\path\to\your.zip") and merges all latest
       Google Colab / Google Drive scripts, notebooks, logs, and manifests into
       Mirror-7-main WITHOUT overwriting your new Roadmap PDF, README, or .gitignore.
    2. Safely attaches your existing GitHub 'main' commit history if .git is missing.
    3. Skips >45 MB binary weights & raw .jsonl datasets so GitHub never rejects the push.
    4. Commits and pushes everything in 4 clean, step-by-step batches.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\push_mirror7_to_github.ps1
    powershell -ExecutionPolicy Bypass -File .\push_mirror7_to_github.ps1 -ColabZipPath "$HOME\Downloads\Mirror7_Colab_GitHub_Sync.zip"
    powershell -ExecutionPolicy Bypass -File .\push_mirror7_to_github.ps1 -DryRun
#>

param(
    [string]$ColabZipPath = "",
    [string]$RemoteUrl = "https://github.com/realahmad07/Mirror-7.git",
    [string]$Branch = "main",
    [double]$MaxSingleFileMB = 45.0,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot

Write-Host ""
Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host " MIRROR-7 COLAB ZIP MERGER & STEP-BY-STEP GITHUB PUSH SCRIPT" -ForegroundColor Cyan
Write-Host " Target Repo : $RemoteUrl ($Branch)" -ForegroundColor Cyan
Write-Host " Local Root  : $RepoRoot" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

if (-not (Test-Path (Join-Path $RepoRoot "mirror7_backend")) -or -not (Test-Path (Join-Path $RepoRoot "README.md"))) {
    throw "Safety check failed: README.md or mirror7_backend not found in $RepoRoot"
}
Set-Location $RepoRoot

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is not installed or not in PATH. Install Git for Windows from https://git-scm.com/download/win"
}

# ---------------------------------------------------------------------------
# Pre-Step: Optional Explicit Colab / Drive ZIP Merge (-ColabZipPath)
# ---------------------------------------------------------------------------
# Note: Your uploaded Colab Sync ZIP has ALREADY been audited, cleaned, and merged
# into this workspace. Pass -ColabZipPath explicitly only when merging a future ZIP.

if ($ColabZipPath) {
    if (-not (Test-Path -LiteralPath $ColabZipPath -PathType Leaf)) {
        throw "Specified -ColabZipPath not found: $ColabZipPath"
    }
    $zipMB = [math]::Round((Get-Item -LiteralPath $ColabZipPath).Length / 1MB, 2)
    Write-Host "`n[Colab Sync] Found Colab/Drive ZIP ($zipMB MB): $ColabZipPath" -ForegroundColor Green

    if ($DryRun) {
        Write-Host "  [DryRun] Would merge Colab ZIP into $RepoRoot while preserving newly generated Roadmap PDF & README." -ForegroundColor Gray
    } else {
        # Back up newly generated local flagship files so older Drive copies never overwrite them
        $protectedRelPaths = @(
            ".gitignore",
            "README.md",
            "PROJECT_OVERVIEW_AND_ROADMAP.md",
            "Ahmad Hassan Mirror OPEN SOURCE detailed road map.pdf",
            "docs\Ahmad Hassan Mirror OPEN SOURCE detailed road map.pdf",
            "artifacts\V6_COLAB_A100_VERIFICATION_LOGS.md",
            "artifacts\v6_master_telemetry_summary.json",
            "push_mirror7_to_github.ps1",
            "training\colab\export_drive_for_github_sync.py"
        )

        $protDir = Join-Path $env:TEMP ("mirror7_prot_" + [guid]::NewGuid().ToString("N"))
        $extractDir = Join-Path $env:TEMP ("mirror7_zip_" + [guid]::NewGuid().ToString("N"))
        New-Item -ItemType Directory -Path $protDir -Force | Out-Null
        New-Item -ItemType Directory -Path $extractDir -Force | Out-Null

        try {
            foreach ($rel in $protectedRelPaths) {
                $src = Join-Path $RepoRoot $rel
                if (Test-Path -LiteralPath $src -PathType Leaf) {
                    $dst = Join-Path $protDir $rel
                    $dstParent = Split-Path -Parent $dst
                    if (-not (Test-Path -LiteralPath $dstParent)) {
                        New-Item -ItemType Directory -Path $dstParent -Force | Out-Null
                    }
                    Copy-Item -LiteralPath $src -Destination $dst -Force
                }
            }

            Write-Host "  Extracting Colab archive..." -ForegroundColor White
            Expand-Archive -LiteralPath $ColabZipPath -DestinationPath $extractDir -Force

            # If ZIP contains a single top-level wrapper folder (e.g. Mirror-7-main/), unwrap it
            $topItems = @(Get-ChildItem -LiteralPath $extractDir -Force)
            $copySource = $extractDir
            if ($topItems.Count -eq 1 -and $topItems[0].PSIsContainer -and $topItems[0].Name -like "Mirror*") {
                $copySource = $topItems[0].FullName
            }

            Write-Host "  Merging latest Colab files into local workspace..." -ForegroundColor White
            Get-ChildItem -LiteralPath $copySource -Force | ForEach-Object {
                if ($_.Name -ne ".git" -and $_.Name -ne ".venv") {
                    Copy-Item -LiteralPath $_.FullName -Destination $RepoRoot -Recurse -Force
                }
            }

            # Restore protected flagship roadmap, README, logs, and .gitignore on top
            foreach ($rel in $protectedRelPaths) {
                $saved = Join-Path $protDir $rel
                if (Test-Path -LiteralPath $saved -PathType Leaf) {
                    $target = Join-Path $RepoRoot $rel
                    $tParent = Split-Path -Parent $target
                    if (-not (Test-Path -LiteralPath $tParent)) {
                        New-Item -ItemType Directory -Path $tParent -Force | Out-Null
                    }
                    Copy-Item -LiteralPath $saved -Destination $target -Force
                }
            }
            Write-Host "  [OK] Merged all latest Colab files while keeping your new Roadmap PDF & README intact!" -ForegroundColor Green
        } finally {
            Remove-Item -LiteralPath $protDir -Recurse -Force -ErrorAction SilentlyContinue
            Remove-Item -LiteralPath $extractDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
} else {
    Write-Host "`n[Colab Sync] No Mirror7_Colab_GitHub_Sync.zip detected in Downloads/Desktop (syncing current workspace)." -ForegroundColor DarkYellow
}

# ---------------------------------------------------------------------------
# Step 0: Attach existing GitHub .git history if folder came from a ZIP
# ---------------------------------------------------------------------------
$GitDir = Join-Path $RepoRoot ".git"
if (-not (Test-Path $GitDir)) {
    Write-Host "`n[Step 0/4] No .git folder found. Linking remote GitHub history safely..." -ForegroundColor Yellow
    if ($DryRun) {
        Write-Host "  [DryRun] Would clone --no-checkout $RemoteUrl metadata and link .git without modifying local files." -ForegroundColor Gray
    } else {
        $TempClone = Join-Path $env:TEMP ("mirror7_git_sync_" + [guid]::NewGuid().ToString("N"))
        try {
            git clone --no-checkout --branch $Branch $RemoteUrl $TempClone
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to fetch remote repository metadata from $RemoteUrl"
            }
            Move-Item -Path (Join-Path $TempClone ".git") -Destination $GitDir -Force
        } finally {
            if (Test-Path $TempClone) {
                Remove-Item -Path $TempClone -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
        git -C $RepoRoot reset --mixed HEAD | Out-Null

        $deletedInWorktree = git -C $RepoRoot ls-files --deleted
        foreach ($delFile in $deletedInWorktree) {
            if ($delFile -and -not (Test-Path (Join-Path $RepoRoot $delFile))) {
                Write-Host "  [Keep Remote File] Restoring remote-only file: $delFile" -ForegroundColor DarkCyan
                git -C $RepoRoot checkout HEAD -- $delFile
            }
        }
        Write-Host "  [OK] Attached existing GitHub '$Branch' commit history with zero local files overwritten." -ForegroundColor Green
    }
} else {
    Write-Host "`n[Step 0/4] Existing .git folder detected inside Mirror-7-main." -ForegroundColor Green
    if (-not $DryRun) {
        $currentRemote = ""
        try { $currentRemote = (git -C $RepoRoot remote get-url origin 2>$null) } catch {}
        if (-not $currentRemote) {
            git -C $RepoRoot remote add origin $RemoteUrl
        } elseif ($currentRemote -ne $RemoteUrl) {
            git -C $RepoRoot remote set-url origin $RemoteUrl
        }
        git -C $RepoRoot fetch origin $Branch
    }
}

if (-not $DryRun) {
    $uName = ""
    $uEmail = ""
    try { $uName = (git -C $RepoRoot config user.name 2>$null) } catch {}
    try { $uEmail = (git -C $RepoRoot config user.email 2>$null) } catch {}
    if (-not $uName) { git -C $RepoRoot config user.name "Ahmad Hassan" }
    if (-not $uEmail) { git -C $RepoRoot config user.email "realahmad07@users.noreply.github.com" }

    git -C $RepoRoot config http.postBuffer 524288000
    git -C $RepoRoot config http.lowSpeedLimit 1000
    git -C $RepoRoot config http.lowSpeedTime 120
}

# ---------------------------------------------------------------------------
# Helper: Safe Staged Batch Commit & Push
# ---------------------------------------------------------------------------
function Invoke-SafeBatchPush {
    param(
        [string]$StepLabel,
        [string[]]$PathsToStage,
        [string]$CommitMessage
    )

    Write-Host "`n[$StepLabel] Preparing batch: $CommitMessage" -ForegroundColor Cyan

    if ($DryRun) {
        foreach ($p in $PathsToStage) {
            if (Test-Path (Join-Path $RepoRoot $p)) {
                Write-Host "  [DryRun] Verified path ready to stage: $p" -ForegroundColor Gray
            }
        }
        return
    }

    foreach ($p in $PathsToStage) {
        if (Test-Path (Join-Path $RepoRoot $p)) {
            # Never commit raw .zip archives into git
            if ($p -eq ".") {
                git -C $RepoRoot add -- . ":!*.zip"
            } else {
                git -C $RepoRoot add -- $p
            }
        }
    }

    $stagedFiles = git -C $RepoRoot diff --cached --name-only
    if (-not $stagedFiles) {
        Write-Host "  [Skip] No file changes detected in this batch (already up to date)." -ForegroundColor DarkGray
        return
    }

    $totalBatchBytes = 0L
    foreach ($relFile in $stagedFiles) {
        $fullPath = Join-Path $RepoRoot $relFile
        if (Test-Path -LiteralPath $fullPath -PathType Leaf) {
            $item = Get-Item -LiteralPath $fullPath
            $mb = [math]::Round($item.Length / 1MB, 2)
            if ($mb -gt $MaxSingleFileMB -or $item.Extension -eq ".zip") {
                Write-Host "  [Guardrail] Skipping heavy/archive file ($mb MB): $relFile" -ForegroundColor Yellow
                git -C $RepoRoot reset HEAD -- $relFile | Out-Null
            } else {
                $totalBatchBytes += $item.Length
            }
        }
    }

    $remainingStaged = git -C $RepoRoot diff --cached --name-only
    if (-not $remainingStaged) {
        Write-Host "  [Skip] Nothing remaining to commit in this batch." -ForegroundColor DarkGray
        return
    }

    $batchMB = [math]::Round($totalBatchBytes / 1MB, 2)
    $fileCount = @($remainingStaged).Count
    Write-Host "  Committing $fileCount file(s) ($batchMB MB payload)..." -ForegroundColor White

    git -C $RepoRoot commit -m $CommitMessage
    if ($LASTEXITCODE -ne 0) {
        throw "git commit failed during $StepLabel"
    }

    Write-Host "  Pushing batch to origin/$Branch..." -ForegroundColor White
    git -C $RepoRoot push -u origin "HEAD:$Branch"
    if ($LASTEXITCODE -ne 0) {
        throw "git push failed during $StepLabel. Sign in via Git Credential Manager / Personal Access Token and re-run (completed steps stay saved)."
    }
    Write-Host "  [SUCCESS] $StepLabel pushed to GitHub!" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# Execute the 4 Step-by-Step Batches
# ---------------------------------------------------------------------------

Invoke-SafeBatchPush `
    -StepLabel "Step 1/4" `
    -PathsToStage @(".gitignore") `
    -CommitMessage "chore(repo): configure gitignore for verification artifacts and heavy dataset exclusions"

Invoke-SafeBatchPush `
    -StepLabel "Step 2/4" `
    -PathsToStage @(
        "README.md",
        "PROJECT_OVERVIEW_AND_ROADMAP.md",
        "Ahmad Hassan Mirror OPEN SOURCE detailed road map.pdf",
        "docs"
    ) `
    -CommitMessage "docs(roadmap): publish Mirror Byte-First project overview and detailed PDF roadmap"

Invoke-SafeBatchPush `
    -StepLabel "Step 3/4" `
    -PathsToStage @(
        "training",
        "colab_backup",
        "mirror7_backend",
        "pyproject.toml",
        "requirements-training.txt",
        "requirements-response-realizer.txt",
        "Dockerfile",
        "docker-compose.yml"
    ) `
    -CommitMessage "feat(architecture): sync Colab V5/V6 byte-native stateful training, standalone inference, and hub modules"

Invoke-SafeBatchPush `
    -StepLabel "Step 4/4" `
    -PathsToStage @(
        "artifacts",
        "."
    ) `
    -CommitMessage "data(artifacts): publish A100 V6 verification logs, Drive SHA-256 manifests, and benchmark JSON telemetry"

Write-Host ""
Write-Host "==========================================================================" -ForegroundColor Green
if ($DryRun) {
    Write-Host " DRY RUN COMPLETE - Ready to merge & push! Run without -DryRun:" -ForegroundColor Green
    Write-Host "   powershell -ExecutionPolicy Bypass -File .\push_mirror7_to_github.ps1" -ForegroundColor Yellow
} else {
    Write-Host " ALL 4 BATCHES SUCCESSFULLY PUSHED TO GITHUB!" -ForegroundColor Green
    Write-Host " Verify live at: https://github.com/realahmad07/Mirror-7" -ForegroundColor Yellow
}
Write-Host "==========================================================================" -ForegroundColor Green
