#!/usr/bin/env python3
"""Deteccao de EPI (CAPACETE e COLETE) com YOLOv8/YOLO NCNN, torch-free.

Roda em CPU ARM (Raspberry Pi 4) usando apenas ncnn + numpy + opencv.
Replica o pre/pos-processamento do Ultralytics: letterbox(114) -> BGR2RGB -> /255
-> ncnn -> decode anchor-free (cx,cy,w,h ja em px, classes ja com sigmoid)
-> NMS por classe (cv2.dnn) -> filtra so capacete/colete.

Model-agnostico: descobre os indices de capacete/colete pelos NOMES das classes
no metadata.yaml (helmet/hardhat -> capacete; vest -> colete; ignora classes "no-*").

Uso:
  python infer_ppe.py                         # imagem de exemplo (URL)
  python infer_ppe.py imagem.jpg              # caminho local
  python infer_ppe.py https://.../foto.jpg    # URL
  EPI_MODEL_DIR=/caminho/modelo python infer_ppe.py foto.jpg
"""
import argparse
import os
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import ncnn
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("EPI_MODEL_DIR", ROOT / "models" / "ppe_ncnn_model"))

DEFAULT_URL = "https://commons.wikimedia.org/wiki/Special:FilePath/Highvis.jpg"


def configure_targets(model_dir):
    """Le metadata.yaml e mapeia indices -> 'capacete'/'colete' pelos nomes das classes.
    Retorna (target {idx: label}, colors {idx: bgr})."""
    names = yaml.safe_load(open(model_dir / "metadata.yaml"))["names"]
    target, colors = {}, {}
    for i, raw in names.items():
        n = str(raw).lower()
        neg = any(n.startswith(p) for p in ("no-", "no_", "no ")) or "nohelmet" in n or "novest" in n
        if neg:
            continue
        if ("helmet" in n or "hardhat" in n or "hard_hat" in n) and "head_no" not in n:
            target[int(i)] = "capacete"; colors[int(i)] = (0, 165, 255)   # laranja
        elif "vest" in n or "colete" in n:
            target[int(i)] = "colete"; colors[int(i)] = (0, 255, 0)       # verde
    return target, colors


def find_no_helmet_id(model_dir):
    """Indice da classe 'sem capacete' (NO-Hardhat), ou None se o modelo nao tiver.
    Usado no controle de acesso: presenca de NO-Hardhat => negar."""
    names = yaml.safe_load(open(model_dir / "metadata.yaml"))["names"]
    for i, raw in names.items():
        n = str(raw).lower()
        neg = any(n.startswith(p) for p in ("no-", "no_", "no "))
        if neg and ("hardhat" in n or "helmet" in n or "hard_hat" in n):
            return int(i)
    return None


def load_net(model_dir, threads):
    net = ncnn.Net()
    net.opt.num_threads = threads
    net.opt.use_vulkan_compute = False
    net.load_param(str(model_dir / "model.ncnn.param"))
    net.load_model(str(model_dir / "model.ncnn.bin"))
    return net


def fetch_image(arg):
    if arg and not arg.startswith("http"):
        return cv2.imread(arg), Path(arg).name
    url = arg or DEFAULT_URL
    req = urllib.request.Request(url, headers={"User-Agent": "epi-pi/1.0 (educational)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), url.split("/")[-1]


def letterbox(img, new, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nw, nh = round(w * r), round(h * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = new - nw, new - nh
    top, left = dh // 2, dw // 2
    out = cv2.copyMakeBorder(resized, top, dh - top, left, dw - left,
                             cv2.BORDER_CONSTANT, value=color)
    return out, r, left, top


def infer(net, img_bgr, imgsz):
    lb, r, padx, pady = letterbox(img_bgr, imgsz)
    mat_in = ncnn.Mat.from_pixels(np.ascontiguousarray(lb),
                                  ncnn.Mat.PixelType.PIXEL_BGR2RGB, imgsz, imgsz)
    mat_in.substract_mean_normalize([0.0, 0.0, 0.0], [1 / 255.0] * 3)
    ex = net.create_extractor()
    ex.input("in0", mat_in)
    t0 = time.perf_counter()
    _, out = ex.extract("out0")
    dt = (time.perf_counter() - t0) * 1000.0
    return np.array(out), dt, r, padx, pady


def decode(preds, r, padx, pady, orig, conf_thres, iou_thres, target_ids):
    preds = np.squeeze(preds).astype(np.float32)
    if preds.shape[0] > preds.shape[1]:          # garante (C, N) com C pequeno
        preds = preds.T
    boxes = preds[:4].T                           # cx,cy,w,h (px do letterbox)
    scores = preds[4:].T                          # (N, nc) ja com sigmoid
    cls_ids = scores.argmax(1)
    confs = scores.max(1)
    keep = (confs > conf_thres) & np.isin(cls_ids, target_ids)   # so capacete/colete
    boxes, confs, cls_ids = boxes[keep], confs[keep], cls_ids[keep]
    if len(boxes) == 0:
        return []
    cx, cy, w, h = boxes.T
    x1, y1 = cx - w / 2, cy - h / 2
    rects = np.stack([x1, y1, w, h], 1).tolist()
    try:
        idxs = cv2.dnn.NMSBoxesBatched(rects, confs.tolist(), cls_ids.tolist(),
                                       conf_thres, iou_thres)
    except AttributeError:
        idxs = cv2.dnn.NMSBoxes(rects, confs.tolist(), conf_thres, iou_thres)
    H, W = orig.shape[:2]
    dets = []
    for i in np.array(idxs).flatten():
        ax1 = max(0, min(W, (x1[i] - padx) / r)); ay1 = max(0, min(H, (y1[i] - pady) / r))
        ax2 = max(0, min(W, (x1[i] + w[i] - padx) / r)); ay2 = max(0, min(H, (y1[i] + h[i] - pady) / r))
        dets.append((int(cls_ids[i]), float(confs[i]), (ax1, ay1, ax2, ay2)))
    return dets


def main():
    ap = argparse.ArgumentParser(description="Detector de EPI (capacete + colete) YOLO NCNN torch-free")
    ap.add_argument("image", nargs="?", default=None, help="caminho local ou URL (default: exemplo)")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "out_ppe.jpg"))
    args = ap.parse_args()

    if not MODEL_DIR.exists():
        raise SystemExit(f"ERRO: modelo NCNN nao encontrado em {MODEL_DIR}\n"
                         f"Defina EPI_MODEL_DIR ou coloque os arquivos em models/ppe_ncnn_model/")

    target, colors = configure_targets(MODEL_DIR)
    if not target:
        raise SystemExit("ERRO: nao encontrei classes de capacete/colete no metadata do modelo")
    target_ids = np.array(sorted(target))
    print(f"[modelo] {MODEL_DIR.name} | classes-alvo: " +
          ", ".join(f"{i}={target[i]}" for i in target_ids), flush=True)

    net = load_net(MODEL_DIR, args.threads)
    img, name = fetch_image(args.image)
    if img is None:
        raise SystemExit("ERRO: nao consegui carregar a imagem")
    print(f"[img] {name} {img.shape[1]}x{img.shape[0]}", flush=True)

    for _ in range(args.warmup):
        infer(net, img, args.imgsz)
    times, last = [], None
    for _ in range(args.runs):
        out, ms, r, px, py = infer(net, img, args.imgsz)
        times.append(ms); last = (out, r, px, py)
    dets = decode(*last, img, args.conf, args.iou, target_ids)

    t = np.array(times)
    print("\n===== EPI (capacete + colete) | NCNN torch-free =====")
    print(f"imgsz={args.imgsz} | threads={args.threads} | CPU")
    print(f"inferencia ncnn (ms): media={t.mean():.1f} min={t.min():.1f} max={t.max():.1f} mediana={np.median(t):.1f}")
    print(f"throughput: ~{1000.0 / t.mean():.1f} FPS (so a rede)")
    n_cap = sum(1 for c, _, _ in dets if target.get(c) == "capacete")
    n_col = sum(1 for c, _, _ in dets if target.get(c) == "colete")
    print(f"deteccoes (conf>{args.conf}): {len(dets)}  ->  capacete: {n_cap} | colete: {n_col}")
    for c, conf, (a, b, x, y) in sorted(dets, key=lambda d: -d[1]):
        print(f"  {target[c]:9s} {conf:.2f}  bbox=({a:.0f},{b:.0f},{x:.0f},{y:.0f})")

    vis = img.copy()
    for c, conf, (a, b, x, y) in dets:
        cv2.rectangle(vis, (int(a), int(b)), (int(x), int(y)), colors[c], 2)
        cv2.putText(vis, f"{target[c]} {conf:.2f}", (int(a), int(b) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[c], 2)
    cv2.imwrite(args.out, vis)
    print(f"[out] imagem anotada salva em {args.out}")


if __name__ == "__main__":
    main()
