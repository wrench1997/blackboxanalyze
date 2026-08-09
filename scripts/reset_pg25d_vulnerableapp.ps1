$ErrorActionPreference = 'Stop'

$pg25ContainerName = 'pg25-vulnerableapp'
$pg25NetworkName = 'pg25-vulnerableapp-hostonly'
$pg25ImageRef = 'sasanlabs/owasp-vulnerableapp@sha256:7bc084dac341f089c6e788d2369a27f599c902d742c5e113d7bb50661cd92406'

$pg25Existing = docker ps -a --filter "name=^/$pg25ContainerName$" --format '{{.Names}}'
if ($pg25Existing -eq $pg25ContainerName) {
    docker stop --time 20 $pg25ContainerName | Out-Null
    docker rm $pg25ContainerName | Out-Null
}

$pg25NetworkPresent = docker network ls --filter "name=^$pg25NetworkName$" --format '{{.Name}}'
if ($pg25NetworkPresent -ne $pg25NetworkName) {
    docker network create --driver bridge `
        --opt com.docker.network.bridge.enable_ip_masquerade=false `
        --opt com.docker.network.bridge.enable_icc=false `
        $pg25NetworkName | Out-Null
}

docker run -d `
    --name $pg25ContainerName `
    --network $pg25NetworkName `
    --publish 127.0.0.1:19090:9090 `
    --read-only `
    --cap-drop ALL `
    --security-opt no-new-privileges `
    --pids-limit 256 `
    --memory 1g `
    --tmpfs /tmp:rw,noexec,nosuid,size=256m `
    --tmpfs /run:rw,noexec,nosuid,size=16m `
    --tmpfs /app/resources/static/upload:rw,noexec,nosuid,size=64m `
    --tmpfs /contentDispositionUpload:rw,noexec,nosuid,size=64m `
    --restart no `
    $pg25ImageRef | Out-Null

$pg25HealthReady = $false
$pg25LastHealth = ''
for ($pg25Attempt = 1; $pg25Attempt -le 30; $pg25Attempt++) {
    $pg25PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'SilentlyContinue'
    try {
        $pg25HealthOutput = docker exec $pg25ContainerName curl --fail --silent --show-error --max-time 5 `
            --output /dev/null --write-out 'status=%{http_code} content_type=%{content_type} url=%{url_effective}\n' `
            http://127.0.0.1:9090/VulnerableApp/ 2>$null
        $pg25HealthExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $pg25PreviousErrorActionPreference
    }
    $pg25LastHealth = ($pg25HealthOutput -join ' ').Trim()
    if ($pg25HealthExitCode -eq 0 -and $pg25LastHealth -match 'status=2\d\d') {
        $pg25HealthReady = $true
        Write-Output $pg25LastHealth
        break
    }
    if ($pg25Attempt -lt 30) {
        Start-Sleep -Seconds 2
    }
}
if (-not $pg25HealthReady) {
    throw "VulnerableApp did not become healthy after 30 attempts. last_health=$pg25LastHealth"
}
