$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$ocrRoot = Join-Path $root "DeepLearning_OCR"
$header = Get-Content -Raw (Join-Path $ocrRoot "DeepLearning_OCR.h")
$source = Get-Content -Raw (Join-Path $ocrRoot "DeepLearning_OCR.cpp")
$interface = Get-Content -Raw (Join-Path $ocrRoot "DeepLearning_OCR_interface.cpp")

if ($header -notmatch 'static\s+void\s+configureOcrLoggingForRequest\s*\(\s*const char\*\s+xmlIn\s*,\s*const std::string&\s+apiName\s*\)\s*;') {
    throw "DeepLearning_OCR.h must declare public static configureOcrLoggingForRequest."
}
if ($header -notmatch 'static\s+bool\s+addOcrLog\s*\(\s*long\s+logTypeCode\s*,\s*const char\*\s+content\s*\)\s*;') {
    throw "DeepLearning_OCR.h must declare public static addOcrLog."
}
if ($source -notmatch 'void\s+DeepLearning_OCR::configureOcrLoggingForRequest') {
    throw "DeepLearning_OCR.cpp must define configureOcrLoggingForRequest."
}
if ($source -notmatch 'bool\s+DeepLearning_OCR::addOcrLog') {
    throw "DeepLearning_OCR.cpp must define addOcrLog."
}
if ($source -notmatch 'debugProcess\s*==\s*1\s*\|\|\s*debugResult\s*==\s*1') {
    throw "OCR logging must use the debugProcess/debugResult OR rule."
}

$directWrites = [regex]::Matches($source, 'g_log\.AddLog\s*\(')
$guardedWrite = 'bool\s+DeepLearning_OCR::addOcrLog\s*\([^)]*\)\s*\{(?s:.*?)return\s+g_log\.AddLog\s*\('
if ($directWrites.Count -ne 1 -or $source -notmatch $guardedWrite) {
    throw "DeepLearning_OCR.cpp must write to g_log only through addOcrLog."
}

if ([regex]::Matches($source, '(?m)^\s*CFileLogEx\s+g_log\s*;').Count -ne 1) {
    throw "DeepLearning_OCR.cpp must contain the single OCR g_log definition."
}
if ([regex]::Matches($source, '(?m)^\s*std::once_flag\s+g_ocrLoggerInitFlag\s*;').Count -ne 1) {
    throw "DeepLearning_OCR.cpp must contain the single OCR logger once_flag definition."
}
if ($interface -match '(?m)^\s*CFileLogEx\s+g_log\s*;') {
    throw "DeepLearning_OCR_interface.cpp must not define g_log."
}
if ($interface -match 'ensureOcrLoggerInitialized') {
    throw "DeepLearning_OCR_interface.cpp must not initialize logging unconditionally."
}

foreach ($api in @("DLE_ocr_init", "DLE_ocr", "DLE_ocr_clean")) {
    $pattern = "int\s+$api\s*\([^)]*\)\s*\{(?s:.*?)DeepLearning_OCR::configureOcrLoggingForRequest\(xmlIn,\s*apiName\)"
    if ($interface -notmatch $pattern) {
        throw "$api must configure OCR logging for the request."
    }
}

Write-Output "OCR log gating source checks passed."
