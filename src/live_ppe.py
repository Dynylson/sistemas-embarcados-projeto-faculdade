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
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from infer_ppe import load_net, infer, decode, configure_targets, MODEL_DIR


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
        dets = decode(out, r, px, py, frame, args.conf, args.iou, target_ids)
        for c, conf, (a, b, x, y) in dets:
            cv2.rectangle(frame, (int(a), int(b)), (int(x), int(y)), colors[c], 2)
            cv2.putText(frame, f"{target[c]} {conf:.2f}", (int(a), int(b) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[c], 2)
        now = time.perf_counter()
        dt = now - t_prev
        t_prev = now
        inst = 1.0 / dt if dt > 0 else 0.0
        ema = inst if ema is None else 0.9 * ema + 0.1 * inst
        n_cap = sum(1 for c, _, _ in dets if target.get(c) == "capacete")
        n_col = sum(1 for c, _, _ in dets if target.get(c) == "colete")
        cv2.putText(frame, f"FPS:{ema:4.1f}  inf:{ms:4.0f}ms  capacete:{n_cap}  colete:{n_col}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok2:
            with state.lock:
                state.jpeg = buf.tobytes()
                state.fps = ema
    cap.release()


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
