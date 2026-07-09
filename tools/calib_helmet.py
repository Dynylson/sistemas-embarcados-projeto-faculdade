#!/usr/bin/env python3
"""Calibracao da autorizacao POR COR do capacete (independente da classe Hardhat,
que o modelo generico confunde). Mede, na regiao da cabeca (topo do box Person),
quanto dos pixels caem na faixa de cor do capacete (HSV inRange).

Captura N frames, usa o ULTIMO, e reporta:
  - HSV mediano da regiao da cabeca
  - fracao de pixels 'cor-capacete' (fill%) com a faixa dada
Salva um comparativo: frame anotado + mascara.

Uso (no Pi, servico epi-live PARADO):
  EPI_MODEL_DIR=~/epi/ppe_ncnn_model python calib_helmet.py \
      --hlo 90 --hhi 118 --smin 50 --vmin 40 --out ~/epi/calib.jpg
"""
import argparse

import cv2
import numpy as np
import yaml

from infer_ppe import load_net, infer, decode, MODEL_DIR


def head_region(person_box, frac=0.42):
    a, b, x, y = person_box
    return int(a), int(b), int(x), int(b + (y - b) * frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--frames", type=int, default=15)
    # faixa de cor candidata (HSV OpenCV: H 0-179)
    ap.add_argument("--hlo", type=int, default=90)
    ap.add_argument("--hhi", type=int, default=118)
    ap.add_argument("--smin", type=int, default=50)
    ap.add_argument("--vmin", type=int, default=40)
    ap.add_argument("--headfrac", type=float, default=0.42)
    ap.add_argument("--out", default="calib.jpg")
    args = ap.parse_args()

    names = yaml.safe_load(open(MODEL_DIR / "metadata.yaml"))["names"]
    names = {int(k): str(v) for k, v in names.items()}
    all_ids = np.array(sorted(names))
    person_id = next((i for i, n in names.items() if n.lower() == "person"), None)

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
        raise SystemExit("ERRO: sem frame")

    out, ms, r, px, py = infer(net, frame, args.imgsz)
    dets = decode(out, r, px, py, frame, args.conf, args.iou, all_ids)
    persons = [d for d in dets if d[0] == person_id]
    if not persons:
        raise SystemExit("ERRO: nenhuma Person detectada; chegue mais perto/enquadre.")
    # maior pessoa (mais proxima)
    _, pconf, pbox = max(persons, key=lambda d: (d[2][2] - d[2][0]) * (d[2][3] - d[2][1]))
    hx1, hy1, hx2, hy2 = head_region(pbox, args.headfrac)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    head = hsv[hy1:hy2, hx1:hx2]
    lo = np.array([args.hlo, args.smin, args.vmin])
    hi = np.array([args.hhi, 255, 255])
    mask = cv2.inRange(head, lo, hi)
    fill = 100.0 * (mask > 0).mean()
    med = [int(np.median(head[:, :, i])) for i in range(3)]
    print(f"Person conf={pconf:.2f}  regiao-cabeca=({hx1},{hy1},{hx2},{hy2})")
    print(f"HSV mediano cabeca = H={med[0]} S={med[1]} V={med[2]}")
    # percentis (ajudam a separar capacete saturado x fundo lavado)
    for ch, nm in enumerate("HSV"):
        ps = [int(np.percentile(head[:, :, ch], p)) for p in (10, 25, 50, 75, 90)]
        print(f"  {nm} percentis[10,25,50,75,90] = {ps}")
    # varredura de smin (fixando H e V): mostra qual isola o capacete
    print(f"faixa base: H[{args.hlo},{args.hhi}] V>={args.vmin} | varredura de S_min:")
    for smin in (50, 80, 100, 120, 140, 160):
        m = cv2.inRange(head, np.array([args.hlo, smin, args.vmin]),
                        np.array([args.hhi, 255, 255]))
        print(f"    S>={smin:3d} -> fill {100.0*(m>0).mean():5.1f}%")
    print(f">>> FILL 'cor-capacete' na cabeca (S>={args.smin}) = {fill:.1f}%")
    # metrica robusta: apos abrir (mata ruido), area do MAIOR componente conectado.
    # capacete = blob denso (maiorCC alto); bone = pixels espalhados (maiorCC baixo).
    print("open morfologico + maior componente conectado (maiorCC = % da cabeca):")
    for smin in (120, 150):
        m = cv2.inRange(head, np.array([args.hlo, smin, args.vmin]),
                        np.array([args.hhi, 255, 255]))
        for k in (3, 5):
            op = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
            of = 100.0 * (op > 0).mean()
            n, _, stats, _ = cv2.connectedComponentsWithStats(op)
            cc = 100.0 * stats[1:, cv2.CC_STAT_AREA].max() / op.size if n > 1 else 0.0
            print(f"  S>={smin} k={k}: fill_open={of:5.1f}%  maiorCC={cc:5.1f}%")

    # visual: frame anotado (esq) + mascara aplicada (dir)
    vis = frame.copy()
    cv2.rectangle(vis, (int(pbox[0]), int(pbox[1])), (int(pbox[2]), int(pbox[3])), (0, 165, 255), 2)
    cv2.rectangle(vis, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
    cv2.putText(vis, f"fill={fill:.0f}%", (hx1, max(20, hy1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    full_mask = cv2.inRange(hsv, lo, hi)
    masked = cv2.bitwise_and(frame, frame, mask=full_mask)
    cv2.imwrite(args.out, np.hstack([vis, masked]))
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
