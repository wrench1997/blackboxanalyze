[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ManifestPath,
    [string]$Root = (Get-Location).Path
)

$ErrorActionPreference = 'Stop'

$rootPath = (Resolve-Path -LiteralPath $Root).Path
$manifestFullPath = (Resolve-Path -LiteralPath $ManifestPath).Path
$manifest = Get-Content -LiteralPath $manifestFullPath -Raw | ConvertFrom-Json

if ($manifest.schema_version -ne 'pg388-demo-asset-manifest-v1') {
    throw 'Unsupported demo asset manifest schema.'
}
if ($null -eq $manifest.assets -or $manifest.assets.Count -eq 0) {
    throw 'Manifest contains no assets.'
}

$failures = @()
foreach ($asset in $manifest.assets) {
    if ([string]::IsNullOrWhiteSpace($asset.path) -or
        [string]::IsNullOrWhiteSpace($asset.sha256) -or
        $asset.sha256 -notmatch '^[0-9a-fA-F]{64}$') {
        $failures += "invalid manifest entry: $($asset.id)"
        continue
    }

    $candidate = Join-Path $rootPath ([string]$asset.path)
    $resolved = $null
    try { $resolved = (Resolve-Path -LiteralPath $candidate).Path } catch { }
    if ($null -eq $resolved) {
        $failures += "missing: $($asset.path)"
        continue
    }
    $rootPrefix = $rootPath.TrimEnd('\') + '\'
    if (-not $resolved.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase) -and
        $resolved -ne $rootPath) {
        $failures += "path escapes root: $($asset.path)"
        continue
    }

    $item = Get-Item -LiteralPath $resolved
    $actual = (Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($item.Length -ne [int64]$asset.bytes) {
        $failures += "size mismatch: $($asset.path) expected=$($asset.bytes) actual=$($item.Length)"
    }
    if ($actual -ne ([string]$asset.sha256).ToLowerInvariant()) {
        $failures += "sha256 mismatch: $($asset.path) expected=$($asset.sha256) actual=$actual"
    }
}

if ($failures.Count -gt 0) {
    $failures | ForEach-Object { Write-Error $_ }
    exit 1
}

[pscustomobject]@{
    status = 'verified'
    manifest = $manifestFullPath
    asset_count = $manifest.assets.Count
    root = $rootPath
} | ConvertTo-Json -Compress
