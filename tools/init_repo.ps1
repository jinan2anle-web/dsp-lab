#!/usr/bin/env pwsh
<#
.SYNOPSIS
    一条命令把「数字信号处理」项目目录变成 git 仓库。

.DESCRIPTION
    本机当前没有安装 Git 时，脚本会打印安装说明并退出（不崩溃）。
    装好 Git 之后重复运行是安全的：已完成的步骤会被自动跳过。

    完成的动作：
      1. git init -b main
      2. core.quotepath=false（中文文件名正常显示）、core.autocrlf=false（交给 .gitattributes）
      3. user.name / user.email 缺失时补上本地配置
      4. 提交一：仓库骨架（README / .gitignore / .gitattributes / docs / dsp / scripts / tools / report）
      5. 提交二：第一、二周归档（task1_w1 / task1_w2 / data / 课程要求 PDF）
      6. 打印 git log 与 git status

.PARAMETER UserName
    要写入 git 的姓名。省略时用 $env:USERNAME。

.PARAMETER UserEmail
    要写入 git 的邮箱。省略时用「用户名@localhost」，仅用于本地提交。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\tools\init_repo.ps1
#>
[CmdletBinding()]
param(
    [string]$UserName  = '',
    [string]$UserEmail = ''
)

$ErrorActionPreference = 'Stop'

# ---- 定位到项目根目录（本脚本位于 tools\ 下）----
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
Write-Host "项目根目录: $repoRoot" -ForegroundColor Cyan

# 统一用数组传参，避免和 PowerShell 的参数绑定规则打架
function Invoke-Git {
    param([string[]]$GitArgs)
    & git @GitArgs
    if ($LASTEXITCODE -ne 0) {
        throw "git $($GitArgs -join ' ') 失败（退出码 $LASTEXITCODE）"
    }
}

function Test-GitQuiet {
    param([string[]]$GitArgs)
    & git @GitArgs *> $null
    return ($LASTEXITCODE -eq 0)
}

# ---------- 0. 检查 git ----------
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host ''
    Write-Host '未找到 git 命令，仓库暂时无法初始化。' -ForegroundColor Red
    Write-Host '请先安装 Git for Windows: https://git-scm.com/download/win' -ForegroundColor Yellow
    Write-Host '安装时全部保持默认即可；装完关掉终端重新打开，再运行本脚本。' -ForegroundColor Yellow
    Write-Host ''
    Write-Host '说明：当前这台机器没有外网，无法自动下载安装包（已确认 api.github.com 不可达）。' -ForegroundColor DarkGray
    Write-Host '仓库骨架（README / .gitignore / docs / dsp / tools）已经就位，装好 Git 后一条命令即可生效。' -ForegroundColor DarkGray
    exit 1
}

Write-Host ("git 版本: " + (& git --version)) -ForegroundColor Green

# ---------- 1. init ----------
if (Test-Path -LiteralPath (Join-Path $repoRoot '.git')) {
    Write-Host '已存在 .git，跳过 git init' -ForegroundColor DarkGray
}
else {
    if (Test-GitQuiet @('init', '-b', 'main')) {
        Write-Host '已创建仓库（分支 main）' -ForegroundColor Green
    }
    else {
        # 老版本 git 不认识 -b
        Invoke-Git @('init')
        Invoke-Git @('symbolic-ref', 'HEAD', 'refs/heads/main')
        Write-Host '已创建仓库（分支 main，旧版 git 回退路径）' -ForegroundColor Green
    }
}

Invoke-Git @('config', 'core.quotepath', 'false')
Invoke-Git @('config', 'core.autocrlf', 'false')

# ---------- 2. 身份 ----------
$currentName = (& git config user.name) 2>$null
$currentEmail = (& git config user.email) 2>$null

if ([string]::IsNullOrWhiteSpace($currentName)) {
    $nameToUse = if ($UserName) { $UserName } else { $env:USERNAME }
    Invoke-Git @('config', 'user.name', $nameToUse)
    Write-Host "已设置 user.name = $nameToUse（可用 -UserName 覆盖）" -ForegroundColor Yellow
}
if ([string]::IsNullOrWhiteSpace($currentEmail)) {
    $mailToUse = if ($UserEmail) { $UserEmail } else { "$($env:USERNAME)@localhost" }
    Invoke-Git @('config', 'user.email', $mailToUse)
    Write-Host "已设置 user.email = $mailToUse（占位地址，可随时改成真实邮箱）" -ForegroundColor Yellow
}

# ---------- 3. 提交一：骨架 ----------
$skeletonPaths = @('README.md', '.gitignore', '.gitattributes', 'docs', 'dsp', 'scripts', 'tools', 'report')
$existing = @($skeletonPaths | Where-Object { Test-Path -LiteralPath (Join-Path $repoRoot $_) })
if ($existing.Count -gt 0) {
    Invoke-Git (@('add', '--') + $existing)
}

if (Test-GitQuiet @('diff', '--cached', '--quiet')) {
    Write-Host '骨架无变更，跳过提交一' -ForegroundColor DarkGray
}
else {
    Invoke-Git @('commit', '-m', 'chore: 建立仓库骨架与实验二三主线目录')
    Write-Host '提交一完成：仓库骨架' -ForegroundColor Green
}

# ---------- 4. 提交二：归档 ----------
$archivePaths = @('task1_w1', 'task1_w2', 'data', '数字信号处理实验.pdf')
$existingArchive = @($archivePaths | Where-Object { Test-Path -LiteralPath (Join-Path $repoRoot $_) })
if ($existingArchive.Count -gt 0) {
    Invoke-Git (@('add', '--') + $existingArchive)
}

if (Test-GitQuiet @('diff', '--cached', '--quiet')) {
    Write-Host '归档无变更，跳过提交二' -ForegroundColor DarkGray
}
else {
    Invoke-Git @('commit', '-m', 'chore: 归档第一、二周成果与原始录音（音频仅保留 data/A 一份）')
    Write-Host '提交二完成：第一、二周归档' -ForegroundColor Green
}

# ---------- 5. 结果 ----------
Write-Host ''
Write-Host '==== 提交历史 ====' -ForegroundColor Cyan
& git log --oneline
Write-Host ''
Write-Host '==== 当前状态 ====' -ForegroundColor Cyan
& git status --short --branch

Write-Host ''
Write-Host '下一步：' -ForegroundColor Cyan
Write-Host '  git add -A; git commit -m "你的说明"    # 日常提交'
Write-Host '  详细约定见 README.md 第五节'
