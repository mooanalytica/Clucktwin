$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ModuleDir = Split-Path -Parent $ScriptDir
$ContainerRoot = Join-Path $ModuleDir "containers"
$DockerfilePath = Join-Path $ModuleDir "container_related\Dockerfile"
$ImageTag = if ($env:IMAGE_TAG) { $env:IMAGE_TAG } else { "poultry_data_preparation:latest" }
$DockerArchiveTar = Join-Path $ContainerRoot "poultry_data_preparation_docker.tar"

New-Item -ItemType Directory -Force -Path $ContainerRoot | Out-Null

if (Test-Path -LiteralPath $DockerArchiveTar) {
    Remove-Item -LiteralPath $DockerArchiveTar -Force
}

Write-Host "[build] Module dir: $ModuleDir"
Write-Host "[build] Image tag: $ImageTag"
Write-Host "[build] Archive path: $DockerArchiveTar"

docker build -t $ImageTag -f $DockerfilePath $ModuleDir
docker save -o $DockerArchiveTar $ImageTag

Write-Host "[done] Created Docker archive at: $DockerArchiveTar"
