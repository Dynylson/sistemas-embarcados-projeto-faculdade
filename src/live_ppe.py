#!/usr/bin/env python3
"""Detecção de EPI (capacete + colete) AO VIVO via webcam USB, com stream MJPEG.

Captura -> inferência NCNN (torch-free) -> anota -> serve um stream MJPEG por HTTP.
Abra http://<ip-do-pi>:8000/ no navegador de outra máquina para ver ao vivo.

Model-agnostico: descobre os indices de capacete/colete pelos nomes das classes
(ver infer_ppe.configure_targets). Funciona com qualquer modelo NCNN de EPI.

Uso (no Pi):
  EPI_MODEL_DIR=$HOME/epi/ppe_ncnn_model python live_ppe.py
  python live_ppe.py --width 1280 --height 720 --port 8000 --conf 0.4

Para parar: Ctrl+C (ou pkill -f live_ppe.py).
"""
import argparse
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

import yaml

from infer_ppe import load_net, infer, decode, configure_targets, MODEL_DIR, ROOT
from audio_alert import AccessAudio, AUTORIZADO, NEGADO
import helmet_cls


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg = None
        self.fps = 0.0
        self.running = True


state = State()


def capture_loop(args):
    if not MODEL_DIR.exists():
        print(f"ERRO: modelo nao encontrado em {MODEL_DIR} (defina EPI_MODEL_DIR)")
        state.running = False
        return
    target, colors = configure_targets(MODEL_DIR)
    if not target:
        print("ERRO: sem classes de capacete/colete no modelo")
        state.running = False
        return
    target_ids = np.array(sorted(target))
    print(f"[modelo] {MODEL_DIR.name} | alvos: " +
          ", ".join(f"{i}={target[i]}" for i in target_ids), flush=True)

    # Decisao de acesso e por CLASSIFICADOR do capacete (ver helmet_cls), NAO pela
    # classe Hardhat (que o modelo confunde com bone). Precisamos da classe 'Person'
    # p/ recortar a cabeca e classificar capacete_ok x nao.
    names = {int(k): str(v) for k, v in
             yaml.safe_load(open(MODEL_DIR / "metadata.yaml"))["names"].items()}
    person_id = next((i for i, n in names.items() if n.lower() == "person"), None)
    if person_id is None:
        print("[audio] AVISO: modelo sem classe 'Person'; autorizacao indisponivel.",
              flush=True)
    all_ids = target_ids
    if person_id is not None:
        all_ids = np.array(sorted(set(target_ids.tolist()) | {person_id}))

    # Classificador do capacete (YOLOv8-cls via NCNN, torch-free). Decide capacete
    # pelo recorte da cabeca -- robusto a bone (ao contrario de regra de cor).
    clf = None
    if args.helmet_model and os.path.isdir(args.helmet_model):
        clf = helmet_cls.HelmetClassifier(args.helmet_model, threads=max(1, args.threads // 2),
                                          norm=args.helmet_norm)
        print(f"[capacete] classificador NCNN {os.path.basename(args.helmet_model)} "
              f"| norm={args.helmet_norm} | limiar={args.helmet_thresh:.2f}", flush=True)
    else:
        print(f"[capacete] AVISO: classificador nao encontrado em {args.helmet_model}; "
              f"autorizacao desativada (defina EPI_HELMET_MODEL).", flush=True)

    audio = AccessAudio(args.audio_dir, device=args.audio_device, enabled=args.audio,
                        stable_frames=args.audio_stable, min_interval_s=args.audio_interval)
    audio.start()
    if args.audio:
        print(f"[audio] ativo | dir={args.audio_dir} | estabiliza={args.audio_stable} frames",
              flush=True)

    net = load_net(MODEL_DIR, args.threads)
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("ERRO: nao consegui abrir a camera")
        state.running = False
        return
    for _ in range(10):
        cap.read()
    print(f"[cam] {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} | "
          f"imgsz={args.imgsz}, {args.threads} threads, conf={args.conf}", flush=True)

    t_prev = time.perf_counter()
    ema = None
    while state.running:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        out, ms, r, px, py = infer(net, frame, args.imgsz)
        dets = decode(out, r, px, py, frame, args.conf, args.iou, all_ids)

        # Desenha coletes (verde) se o modelo achar; capacete e decidido por cor.
        for c, conf, (a, b, x, y) in dets:
            if target.get(c) == "colete":
                cv2.rectangle(frame, (int(a), int(b)), (int(x), int(y)), (0, 255, 0), 2)
                cv2.putText(frame, f"colete {conf:.2f}", (int(a), int(b) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # AUTORIZACAO POR CLASSIFICADOR: maior pessoa (mais proxima da portaria),
        # recorta a cabeca e classifica capacete_ok x nao.
        persons = [d for d in dets if d[0] == person_id]
        prob = 0.0
        has_helmet = False
        if persons and clf is not None:
            _, _, pbox = max(persons, key=lambda d: (d[2][2] - d[2][0]) * (d[2][3] - d[2][1]))
            crop = helmet_cls.head_crop(frame, pbox)
            if crop is not None:
                prob = clf.prob_capacete(crop)
                has_helmet = prob >= args.helmet_thresh
            a, b, x, y = pbox
            hy2 = int(b + 0.42 * (y - b))
            hcol = (0, 200, 0) if has_helmet else (0, 0, 255)
            cv2.rectangle(frame, (int(a), int(b)), (int(x), hy2), hcol, 2)
            cv2.putText(frame, f"capacete {prob*100:.0f}%", (int(a), max(18, int(b) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, hcol, 2)
        no_helmet = bool(persons) and clf is not None and not has_helmet
        gate_state = audio.update(has_helmet, no_helmet)

        now = time.perf_counter()
        dt = now - t_prev
        t_prev = now
        inst = 1.0 / dt if dt > 0 else 0.0
        ema = inst if ema is None else 0.9 * ema + 0.1 * inst
        cv2.putText(frame, f"FPS:{ema:4.1f}  inf:{ms:4.0f}ms  pessoas:{len(persons)}  capacete:{prob*100:.0f}%",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        banner = {AUTORIZADO: ("ACESSO AUTORIZADO", (0, 200, 0)),
                  NEGADO: ("ACESSO NEGADO", (0, 0, 255))}.get(gate_state)
        if banner:
            txt, col = banner
            cv2.putText(frame, txt, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok2:
            with state.lock:
                state.jpeg = buf.tobytes()
                state.fps = ema
    cap.release()
    audio.stop()


PAGE = (b"<html><head><title>EPI ao vivo</title></head>"
        b"<body style='margin:0;background:#111;text-align:center'>"
        b"<img src='/stream' style='max-width:100%;height:auto'></body></html>")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while state.running:
                    with state.lock:
                        j = state.jpeg
                    if j is None:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(j)}\r\n\r\n".encode())
                    self.wfile.write(j)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()


def main():
    ap = argparse.ArgumentParser(description="EPI (capacete+colete) ao vivo via webcam + MJPEG")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--port", type=int, default=8000)
    # --- audio (controle de acesso por voz) ---
    ap.add_argument("--audio", dest="audio", action="store_true", default=True,
                    help="habilita disparo de voz (padrao)")
    ap.add_argument("--no-audio", dest="audio", action="store_false",
                    help="desabilita o audio")
    ap.add_argument("--audio-dir", default=os.environ.get("EPI_AUDIO_DIR", str(ROOT / "audio")),
                    help="pasta com autorizado.wav / negado.wav (ou defina EPI_AUDIO_DIR)")
    ap.add_argument("--audio-device", default=os.environ.get("EPI_AUDIO_DEVICE"),
                    help="dispositivo ALSA do aplay, ex.: plughw:1,0 (default: padrao do sistema)")
    ap.add_argument("--audio-stable", type=int, default=5,
                    help="frames estaveis p/ firmar a decisao antes de falar")
    ap.add_argument("--audio-interval", type=float, default=2.0,
                    help="intervalo minimo (s) entre falas")
    # --- capacete por classificador (autorizacao) ---
    ap.add_argument("--helmet-model",
                    default=os.environ.get("EPI_HELMET_MODEL", str(ROOT / "models" / "helmet_cls_ncnn_model")),
                    help="pasta do modelo YOLOv8-cls NCNN (ou defina EPI_HELMET_MODEL)")
    ap.add_argument("--helmet-thresh", type=float, default=0.5,
                    help="prob minima de 'capacete_ok' p/ autorizar")
    ap.add_argument("--helmet-norm", default="plain", choices=["plain", "imagenet"],
                    help="normalizacao de entrada do classificador (calibrado: plain)")
    args = ap.parse_args()

    th = threading.Thread(target=capture_loop, args=(args,), daemon=True)
    th.start()
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[http] stream em http://<ip-do-pi>:{args.port}/  (Ctrl+C para parar)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        srv.shutdown()


if __name__ == "__main__":
    main()
