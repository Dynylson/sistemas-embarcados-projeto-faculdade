#!/usr/bin/env python3
"""Diagnostico de calibracao do capacete.

Captura 1 frame da webcam, roda o detector com TODAS as classes e reporta o que
o modelo ve (classe, confianca, bbox) + a cor mediana (HSV) da regiao de cada
'Hardhat'. Salva um frame anotado. Serve para calibrar a autorizacao pelo
capacete ESPECIFICO (impedir que bone passe como capacete).

Uso (no Pi, com o servico epi-live PARADO p/ liberar a camera):
  EPI_MODEL_DIR=~/epi/ppe_ncnn_model python diag_capture.py --out ~/epi/snap.jpg
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from infer_ppe import load_net, infer, decode, MODEL_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25)   # baixo p/ ver marginais
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--out", default="snap.jpg")
    args = ap.parse_args()

    names = yaml.safe_load(open(MODEL_DIR / "metadata.yaml"))["names"]
    names = {int(k): str(v) for k, v in names.items()}
    all_ids = np.array(sorted(names))

    net = load_net(MODEL_DIR, args.threads)
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("ERRO: camera nao abriu (servico epi-live ainda usando /dev/video0?)")
    frame = None
    for _ in range(args.frames):
        ok, frame = cap.read()
    cap.release()
    if frame is None:
        raise SystemExit("ERRO: nao capturei frame")

    out, ms, r, px, py = infer(net, frame, args.imgsz)
    dets = decode(out, r, px, py, frame, args.conf, args.iou, all_ids)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    print(f"deteccoes (conf>{args.conf}): {len(dets)}")
    for c, conf, (a, b, x, y) in sorted(dets, key=lambda d: -d[1]):
        line = f"  {names[c]:14s} {conf:.2f}  bbox=({a:.0f},{b:.0f},{x:.0f},{y:.0f})"
        is_hat = names[c].lower() == "hardhat"
        if is_hat:
            x1, y1, x2, y2 = map(int, (a, b, x, y))
            roi = hsv[y1:y1 + max(1, (y2 - y1) // 2), x1:x2]   # topo do box (calota)
            if roi.size:
                hh, ss, vv = (int(np.median(roi[:, :, i])) for i in range(3))
                line += f"   HSV_topo=(H={hh},S={ss},V={vv})"
        print(line)
        col = (0, 255, 0) if is_hat else (0, 165, 255)
        cv2.rectangle(frame, (int(a), int(b)), (int(x), int(y)), col, 2)
        cv2.putText(frame, f"{names[c]} {conf:.2f}", (int(a), int(b) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    cv2.imwrite(args.out, frame)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
