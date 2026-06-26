$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$toolRoot = Join-Path $root "DeepLearning_AI_tool"
$interface = Get-Content -Raw (Join-Path $toolRoot "DeepLearning_AI_interface.cpp")
$modules = Get-ChildItem $toolRoot -Filter "DeepLearning_AI_*.cpp" |
    Where-Object { $_.Name -notin @("DeepLearning_AI_interface.cpp") }

$logHeader = Join-Path $toolRoot "DeepLearning_AI_log.h"
if (Test-Path $logHeader) {
    throw "DeepLearning_AI_log.h should be removed."
}

$tasks = @(
    @{ Header = "DeepLearning_AI_detect.h"; Source = "DeepLearning_AI_detect.cpp"; Class = "DeepLearning_AI_detect"; Map = "g_mapDLE_AI_detect" },
    @{ Header = "DeepLearning_AI_segment.h"; Source = "DeepLearning_AI_segment.cpp"; Class = "DeepLearning_AI_segment"; Map = "g_mapDLE_AI_segment" },
    @{ Header = "DeepLearning_AI_obb.h"; Source = "DeepLearning_AI_obb.cpp"; Class = "DeepLearning_AI_obb"; Map = "g_mapDLE_AI_obb" },
    @{ Header = "DeepLearning_AI_pose.h"; Source = "DeepLearning_AI_pose.cpp"; Class = "DeepLearning_AI_pose"; Map = "g_mapDLE_AI_pose" }
)

foreach ($task in $tasks) {
    $headerContent = Get-Content -Raw (Join-Path $toolRoot $task.Header)
    $sourceContent = Get-Content -Raw (Join-Path $toolRoot $task.Source)
    if ($headerContent -notmatch 'static\s+void\s+configureAiLoggingForRequest\s*\(\s*const char\*\s+xmlIn\s*,\s*const std::string&\s+apiName\s*\)\s*;') {
        throw "$($task.Header) must declare public static configureAiLoggingForRequest."
    }
    if ($headerContent -notmatch 'static\s+bool\s+addAiLog\s*\(\s*long\s+logTypeCode\s*,\s*const char\*\s+content\s*\)\s*;') {
        throw "$($task.Header) must declare public static addAiLog."
    }
    if ($sourceContent -notmatch ([regex]::Escape("void $($task.Class)::configureAiLoggingForRequest"))) {
        throw "$($task.Source) must define the class-specific configureAiLoggingForRequest."
    }
    if ($sourceContent -notmatch ([regex]::Escape("bool $($task.Class)::addAiLog"))) {
        throw "$($task.Source) must define the class-specific addAiLog."
    }
    $mapDefinition = "std::unordered_map<int,\s*$([regex]::Escape($task.Class))\*>\s*$([regex]::Escape($task.Map))\s*;"
    if ($sourceContent -notmatch $mapDefinition) {
        throw "$($task.Source) must define its own model map."
    }
}

$logHeaderIncludes = Get-ChildItem $toolRoot -Filter "*.cpp" |
    Select-String -Pattern '#include\s+"DeepLearning_AI_log\.h"'
if ($logHeaderIncludes) {
    throw "Source files must not include DeepLearning_AI_log.h."
}

$taskSources = $tasks | ForEach-Object {
    Get-Content -Raw (Join-Path $toolRoot $_.Source)
}
$allTaskSource = $taskSources -join "`n"
$loggerDefinitions = [regex]::Matches($allTaskSource, '(?m)^\s*CFileLogEx\s+g_log\s*;')
$loggerInitFlagDefinitions = [regex]::Matches($allTaskSource, '(?m)^\s*std::once_flag\s+g_aiLoggerInitFlag\s*;')
if ($loggerDefinitions.Count -ne 1) {
    throw "g_log must have exactly one definition; found $($loggerDefinitions.Count)."
}
if ($loggerInitFlagDefinitions.Count -ne 1) {
    throw "g_aiLoggerInitFlag must have exactly one definition; found $($loggerInitFlagDefinitions.Count)."
}

$processApis = @(
    "DLE_AI_detect",
    "DLE_AI_segment",
    "DLE_AI_obb",
    "DLE_AI_pose"
)

foreach ($api in $processApis) {
    $className = switch -Regex ($api) {
        "detect" { "DeepLearning_AI_detect"; break }
        "segment" { "DeepLearning_AI_segment"; break }
        "obb" { "DeepLearning_AI_obb"; break }
        "pose" { "DeepLearning_AI_pose"; break }
    }
    $pattern = "int\s+$api\s*\([^)]*\)\s*\{(?s:.*?)$className::configureAiLoggingForRequest\(xmlIn,\s*apiName\)"
    if ($interface -notmatch $pattern) {
        throw "$api does not configure logging through $className."
    }
}

if ($interface -match '(?m)^(void\s+configureAiLoggingForRequest|bool\s+addAiLog)\s*\(') {
    throw "DeepLearning_AI_interface.cpp must not define the task logging methods."
}
if ($interface -match 'std::unordered_map<int,\s*DeepLearning_AI_') {
    throw "DeepLearning_AI_interface.cpp must not define task model maps."
}

foreach ($task in $tasks) {
    $sourceContent = Get-Content -Raw (Join-Path $toolRoot $task.Source)
    if ($sourceContent -notmatch "debugProcess\s*==\s*1\s*\|\|\s*debugResult\s*==\s*1") {
        throw "$($task.Source) does not use the expected debugProcess/debugResult OR rule."
    }
}

foreach ($task in $tasks) {
    $content = Get-Content -Raw (Join-Path $toolRoot $task.Source)
    $directWrites = [regex]::Matches($content, "g_log\.AddLog\s*\(")
    $guardedWritePattern = "bool\s+$([regex]::Escape($task.Class))::addAiLog\s*\([^)]*\)\s*\{(?s:.*?)return\s+g_log\.AddLog\s*\("
    if ($directWrites.Count -ne 1 -or $content -notmatch $guardedWritePattern) {
        throw "$($task.Source) must write to g_log only through its class-specific addAiLog."
    }
}

Write-Output "AI log gating source checks passed."
