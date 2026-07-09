#!/usr/bin/env python3
"""Disparo de audio para controle de acesso por EPI (portaria).

Ideia: a cada frame o detector diz o que ve (capacete presente / ausente / nada).
Como o loop roda a ~10 FPS, uma decisao "crua" por frame tremeria e o audio
"gaguejaria". Entao aqui ha DUAS pecas:

  1. AudioPlayer  -> toca WAV de forma NAO-bloqueante (thread + fila + aplay).
                     aplay bloqueia ~1-2 s por fala; se tocasse no loop de captura
                     o stream/FPS travaria. Por isso uma thread dedicada consome
                     a fila e frames de audio duplicados sao descartados.

  2. AccessGate   -> maquina de estados com DEBOUNCE + borda + "reset ao sair".
                     - decisao vira estado oficial so apos N frames estaveis;
                     - o audio toca so na TRANSICAO (borda), nunca repetido/frame;
                     - so reanuncia a MESMA decisao depois que a pessoa sai do
                       quadro (estado volta a 'idle'), evitando spam com alguem
                       parado na frente.

Backend de audio: ALSA via `aplay` (alto-falante USB/P2 no Pi). Sem dependencias
Python novas. Os WAV sao gerados no PC (ver tools/gen_audio_windows.ps1).
"""
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

# Estados possiveis do portao.
IDLE = "idle"            # nada relevante no quadro
AUTORIZADO = "autorizado"  # capacete detectado
NEGADO = "negado"          # pessoa sem capacete (NO-Hardhat)


class AudioPlayer:
    """Toca arquivos WAV via `aplay` numa thread separada (nao bloqueia o chamador).

    A fila tem tamanho 1: se ja houver algo tocando/pendente, novos pedidos sao
    descartados (nao faz sentido enfileirar falas antigas de acesso)."""

    def __init__(self, device=None, player="aplay", enabled=True):
        self.device = device        # ex.: "plughw:1,0" para escolher a placa ALSA
        self.player = player
        self.enabled = enabled
        self._q = queue.Queue(maxsize=1)
        self._busy = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        if enabled and shutil.which(player) is None:
            print(f"[audio] AVISO: '{player}' nao encontrado no PATH; audio desativado.",
                  flush=True)
            self.enabled = False

    def start(self):
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stop.is_set():
            try:
                path = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            self._busy.set()
            try:
                cmd = [self.player, "-q"]
                if self.device:
                    cmd += ["-D", self.device]
                cmd.append(str(path))
                subprocess.run(cmd, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, check=False)
            except Exception as e:  # nunca deixar a thread de audio derrubar nada
                print(f"[audio] erro ao tocar {path}: {e}", flush=True)
            finally:
                self._busy.clear()

    def play(self, path):
        """Enfileira um WAV. Retorna False se descartado (ja tocando ou desativado)."""
        if not self.enabled or path is None:
            return False
        if self._busy.is_set():
            return False           # ja esta falando: nao empilha
        try:
            self._q.put_nowait(path)
            return True
        except queue.Full:
            return False

    def stop(self):
        self._stop.set()


class AccessGate:
    """Maquina de estados de acesso com debounce, borda e reset ao sair do quadro.

    Uso por frame:
        gate.update(has_helmet=<bool>, no_helmet=<bool>)
    Retorna o nome do evento a anunciar (AUTORIZADO/NEGADO) SOMENTE no frame da
    transicao; caso contrario None. O estado estavel atual fica em gate.state.
    """

    def __init__(self, stable_frames=5, min_interval_s=2.0):
        self.stable_frames = stable_frames    # frames consecutivos p/ firmar estado
        self.min_interval_s = min_interval_s  # guarda extra anti-repeticao
        self.state = IDLE          # estado estavel atual
        self.announced = IDLE      # ultimo estado que gerou audio
        self._pending = IDLE
        self._count = 0
        self._last_announce_t = -1e9

    @staticmethod
    def _raw_decision(has_helmet, no_helmet):
        # Prioridade a NEGADO por seguranca: se alguem sem capacete aparece, nega.
        if no_helmet:
            return NEGADO
        if has_helmet:
            return AUTORIZADO
        return IDLE

    def update(self, has_helmet, no_helmet, now=None):
        now = time.monotonic() if now is None else now
        raw = self._raw_decision(has_helmet, no_helmet)

        # Debounce: acumula frames estaveis antes de firmar a decisao.
        if raw == self._pending:
            self._count += 1
        else:
            self._pending = raw
            self._count = 1
        if self._count < self.stable_frames:
            return None

        prev = self.state
        self.state = raw
        if raw == prev:
            return None            # nada mudou: nenhuma borda

        # Transicao firmada. Se voltou a IDLE (pessoa saiu), rearma p/ proximo.
        if raw == IDLE:
            self.announced = IDLE
            return None

        # AUTORIZADO / NEGADO. Decisao DIFERENTE da ultima falada sempre anuncia
        # (ex.: negado -> pessoa poe o capacete -> autorizado, mesmo que rapido).
        # So o intervalo minimo trava REPETIR a mesma fala (ex.: oscilacao sem
        # passar por idle); repeticao de pessoa parada ja e barrada por 'announced'.
        if raw == self.announced and (now - self._last_announce_t) < self.min_interval_s:
            return None
        self.announced = raw
        self._last_announce_t = now
        return raw


class AccessAudio:
    """Cola AudioPlayer + AccessGate: recebe deteccoes por frame e toca a fala certa."""

    def __init__(self, audio_dir, device=None, enabled=True,
                 stable_frames=5, min_interval_s=2.0):
        self.dir = Path(audio_dir)
        self.player = AudioPlayer(device=device, enabled=enabled)
        self.gate = AccessGate(stable_frames=stable_frames, min_interval_s=min_interval_s)
        self.wavs = {AUTORIZADO: self.dir / "autorizado.wav",
                     NEGADO: self.dir / "negado.wav"}
        if enabled:
            faltando = [str(p) for p in self.wavs.values() if not p.exists()]
            if faltando:
                print("[audio] AVISO: WAV(s) ausente(s): " + ", ".join(faltando) +
                      " -> gere com tools/gen_audio_windows.ps1 e copie p/ o Pi.",
                      flush=True)

    def start(self):
        self.player.start()

    def update(self, has_helmet, no_helmet):
        """Chame 1x por frame. Toca a fala na transicao. Retorna o estado estavel."""
        event = self.gate.update(has_helmet, no_helmet)
        if event is not None:
            self.player.play(self.wavs.get(event))
        return self.gate.state

    def stop(self):
        self.player.stop()
