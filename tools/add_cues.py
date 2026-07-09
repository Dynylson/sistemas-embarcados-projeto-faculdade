#!/usr/bin/env python3
"""Embute um efeito sonoro (jingle) ANTES da voz em cada WAV de acesso.

Fluxo dos audios:
    tools/gen_audio_windows.ps1  ->  audio/voice/*.wav   (so a voz, SAPI pt-BR)
    tools/add_cues.py            ->  audio/*.wav          (jingle + voz, final)

O Pi usa os arquivos finais em audio/ (autorizado.wav / negado.wav). Como o
jingle fica embutido no mesmo arquivo, o Pi continua tocando 1 arquivo por
evento (sem mudanca no codigo) e o som sempre precede a fala, perfeitamente
sincronizado.

- autorizado: jingle curto ASCENDENTE (agudo, "sucesso").
- negado:     buzzer GRAVE em duas batidas ("erro").

Stdlib pura (wave + math + struct), sem numpy. Formato = o mesmo da voz
(SAPI: 22050 Hz, 16-bit, mono).

Uso:
    python tools/add_cues.py               # voice/ -> audio/
    python tools/add_cues.py <in_dir> <out_dir>
"""
import math
import struct
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAXAMP = 32767


def envelope(i, n, sr, attack=0.006, release=0.060):
    """Ganho 0..1 com ataque/decaimento lineares (evita 'clique' no inicio/fim)."""
    t = i / sr
    dur = n / sr
    a = min(1.0, t / attack) if attack > 0 else 1.0
    r = min(1.0, (dur - t) / release) if release > 0 else 1.0
    return max(0.0, min(a, r))


def tone(freq, dur, sr, vol=0.5, harmonics=(1.0,)):
    """Gera uma nota (soma de harmonicos) com envelope. Retorna lista de int16."""
    n = int(dur * sr)
    norm = sum(abs(h) for h in harmonics) or 1.0
    out = []
    for i in range(n):
        s = 0.0
        for k, amp in enumerate(harmonics, start=1):
            s += amp * math.sin(2 * math.pi * freq * k * (i / sr))
        s = (s / norm) * vol * envelope(i, n, sr)
        out.append(int(max(-1.0, min(1.0, s)) * MAXAMP))
    return out


def silence(dur, sr):
    return [0] * int(dur * sr)


def make_cue(kind, sr):
    """Monta a lista de samples do efeito para 'autorizado' ou 'negado'."""
    if kind == "autorizado":
        # tres notas ascendentes, timbre limpo (sino) => sensacao de "ok/liberado"
        seq = [(880, 0.09), (1175, 0.09), (1760, 0.16)]   # A5 -> D6 -> A6
        buf = []
        for f, d in seq:
            buf += tone(f, d, sr, vol=0.55, harmonics=(1.0, 0.35, 0.12))
        return buf
    # negado: buzzer grave e aspero (harmonicos impares ~ onda quadrada), 2 batidas
    beep = lambda: tone(165, 0.17, sr, vol=0.5, harmonics=(1.0, 0.0, 0.5, 0.0, 0.3))
    return beep() + silence(0.05, sr) + beep()


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        p = w.getparams()
        frames = w.readframes(p.nframes)
    return p, frames


def combine(kind, voice_path, out_path, gap=0.15):
    params, voice = read_wav(voice_path)
    if params.sampwidth != 2 or params.nchannels != 1:
        raise SystemExit(f"ERRO: {voice_path.name} nao e 16-bit mono (esperado da voz SAPI).")
    sr = params.framerate
    cue = make_cue(kind, sr) + silence(gap, sr)
    cue_bytes = struct.pack("<%dh" % len(cue), *cue)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(cue_bytes + voice)
    total_s = (len(cue) + len(voice) // 2) / sr
    print(f"[ok] {kind:10} -> {out_path}  (~{total_s:.1f}s: jingle {len(cue)/sr:.2f}s + voz)")


def main():
    in_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "audio" / "voice"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("autorizado", "negado"):
        src = in_dir / f"{kind}.wav"
        if not src.exists():
            raise SystemExit(f"ERRO: voz nao encontrada: {src}\n"
                             f"Rode antes: tools/gen_audio_windows.ps1 (saida em {in_dir})")
        combine(kind, src, out_dir / f"{kind}.wav")


if __name__ == "__main__":
    main()
