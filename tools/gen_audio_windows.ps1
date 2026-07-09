<#
Gera os WAV de VOZ (pt-BR) para o controle de acesso por EPI, usando a voz
SAPI do Windows (Microsoft Maria Desktop). Rode NO PC.

  powershell -ExecutionPolicy Bypass -File tools\gen_audio_windows.ps1
  python tools\add_cues.py         # embute o jingle antes da voz -> audio\*.wav

Saida deste script: audio\voice\autorizado.wav e audio\voice\negado.wav (so a
voz, PCM 22kHz 16-bit mono). O add_cues.py combina jingle+voz em audio\*.wav,
que sao os arquivos que o Pi realmente toca. Para trocar frases/voz, edite abaixo.
#>
param(
    [string]$VoiceName = "Microsoft Maria Desktop",
    [string]$OutDir    = (Join-Path $PSScriptRoot "..\audio\voice")
)

Add-Type -AssemblyName System.Speech

# frase por arquivo. Criterio escolhido: SO capacete.
$falas = @{
    "autorizado" = "Acesso autorizado."
    "negado"     = "Acesso negado. Coloque o capacete."
}

$OutDir = [System.IO.Path]::GetFullPath($OutDir)
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $synth.SelectVoice($VoiceName)
} catch {
    Write-Warning "Voz '$VoiceName' nao encontrada; usando a voz padrao do sistema."
}
$synth.Rate = 0     # -10 (lento) a +10 (rapido)
$synth.Volume = 100

# PCM 22050 Hz, 16-bit, mono -> compativel com aplay
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono)

foreach ($nome in $falas.Keys) {
    $texto = $falas[$nome]
    $path  = Join-Path $OutDir "$nome.wav"
    $synth.SetOutputToWaveFile($path, $fmt)
    $synth.Speak($texto)
    $synth.SetOutputToNull()
    Write-Host ("[ok] {0,-11} -> {1}  (`"{2}`")" -f $nome, $path, $texto)
}
$synth.Dispose()
Write-Host ""
Write-Host "Voz gerada. Agora embuta o jingle e gere os arquivos finais:"
Write-Host "  python tools\add_cues.py"
Write-Host "Depois copie os finais para o Pi:"
Write-Host "  scp -i `$HOME/.ssh/id_raspberry audio\autorizado.wav audio\negado.wav projeto-embarcados@10.0.0.165:~/epi/audio/"
