#!/usr/bin/env python3
"""Coletor de dataset p/ o classificador do capacete (YOLOv8-cls).

Detecta 'Person' (confiavel) e salva RECORTES QUADRADOS da cabeca em
dataset/<label>/. O rotulo e a pasta -- sem desenhar caixas. Depois isso vai
pro PC treinar (YOLOv8n-cls), exporta NCNN e volta pro Pi.

Rode com o servico epi-live PARADO (libera a camera). Colete variando pose:
vire a cabeca p/ os lados, incline, aproxime/afaste, mude um pouco a luz.

Uso (no Pi):
  # com o capacete:
  EPI_MODEL_DIR=~/epi/ppe_ncnn_model python collect_dataset.py --label capacete_ok --n 100
  # com o bone (acumula na classe 'nao'):
  ... python collect_dataset.py --label nao --n 60
  # sem nada na cabeca (tambem 'nao'):
  ... python collect_dataset.py --label nao --n 40
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from infer_ppe import load_net, infer, decode, MODEL_DIR

ROOT = Path(__file__).resolve().parent


def head_crop(frame, pbox, size=224):
    a, b, x, y = pbox
    ph = y - b
    cx = (a + x) / 2.0
    cy = b + 0.22 * ph
    side = 0.58 * ph
    H, W = frame.shape[:2]
    x1 = max(0, int(cx - side / 2)); y1 = max(0, int(cy - side / 2))
    x2 = min(W, int(cx + side / 2)); y2 = min(H, int(cy + side / 2))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    return cv2.resize(frame[y1:y2, x1:x2], (size, size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="pasta/classe (ex.: capacete_ok, nao)")
    ap.add_argument("--n", type=int, default=100, help="quantos recortes salvar")
    ap.add_argument("--every", type=int, default=2, help="salva 1 a cada N frames")
    ap.add_argument("--outdir", default=str(Path.home() / "epi" / "dataset"))
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    names = {int(k): str(v) for k, v in
             yaml.safe_load(open(MODEL_DIR / "metadata.yaml"))["names"].items()}
    person_id = next((i for i, n in names.items() if n.lower() == "person"), None)
    all_ids = np.array(sorted(names))

    outdir = Path(args.outdir) / args.label
    outdir.mkdir(parents=True, exist_ok=True)
    existing = len(list(outdir.glob("*.jpg")))

    net = load_net(MODEL_DIR, args.threads)
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("ERRO: camera nao abriu (servico epi-live ainda usando /dev/video0?)")
    for _ in range(8):
        cap.read()

    print(f"[coleta] label='{args.label}' meta={args.n} | ja havia {existing} | "
          f"MEXA a cabeca (lados/inclinar/perto-longe)...", flush=True)
    saved, i, miss = 0, 0, 0
    preview = None
    while saved < args.n:
        ok, frame = cap.read()
        if not ok:
            continue
        i += 1
        if i % args.every:
            continue
        out, ms, r, px, py = infer(net, frame, args.imgsz)
        dets = decode(out, r, px, py, frame, args.conf, args.iou, all_ids)
        persons = [d for d in dets if d[0] == person_id]
        if not persons:
            miss += 1
            continue
        _, _, pbox = max(persons, key=lambda d: (d[2][2] - d[2][0]) * (d[2][3] - d[2][1]))
        crop = head_crop(frame, pbox, args.size)
        if crop is None:
            continue
        fname = outdir / f"{args.label}_{existing + saved:04d}.jpg"
        cv2.imwrite(str(fname), crop)
        if preview is None:
            preview = crop
        saved += 1
        if saved % 10 == 0:
            print(f"  {saved}/{args.n}", flush=True)
    cap.release()
    if preview is not None:
        cv2.imwrite(str(Path(args.outdir) / f"_preview_{args.label}.jpg"), preview)
    print(f"[ok] salvei {saved} em {outdir}  (sem-pessoa ignorados: {miss})", flush=True)


if __name__ == "__main__":
    main()
