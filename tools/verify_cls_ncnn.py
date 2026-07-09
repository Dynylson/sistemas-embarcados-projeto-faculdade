#!/usr/bin/env python3
"""Valida o classificador NCNN nos recortes rotulados (roda NO PI, torch-free).

Testa cada normalizacao candidata e reporta acuracia + separacao de score, para
TRAVAR a config certa (normalizacao e limiar) antes de integrar ao live.

Uso (no Pi):
  python verify_cls_ncnn.py --model ~/epi/helmet_cls_ncnn_model --data ~/epi/dataset
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import ncnn

MEANS = {
    "imagenet": ([0.485 * 255, 0.456 * 255, 0.406 * 255],
                 [1 / (0.229 * 255), 1 / (0.224 * 255), 1 / (0.225 * 255)]),
    "plain": ([0, 0, 0], [1 / 255.0] * 3),
}


def infer(net, img, imgsz, mean, norm, inp, outp):
    im = cv2.resize(img, (imgsz, imgsz))
    mat = ncnn.Mat.from_pixels(np.ascontiguousarray(im),
                               ncnn.Mat.PixelType.PIXEL_BGR2RGB, imgsz, imgsz)
    mat.substract_mean_normalize(list(mean), list(norm))
    ex = net.create_extractor()
    ex.input(inp, mat)
    _, out = ex.extract(outp)
    return np.array(out).flatten().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "epi" / "helmet_cls_ncnn_model"))
    ap.add_argument("--data", default=str(Path.home() / "epi" / "dataset"))
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--inp", default="in0")
    ap.add_argument("--outp", default="out0")
    args = ap.parse_args()

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.load_param(str(Path(args.model) / "model.ncnn.param"))
    net.load_model(str(Path(args.model) / "model.ncnn.bin"))

    classes = sorted(p.name for p in Path(args.data).iterdir() if p.is_dir())
    print(f"classes: {classes}  (indice 0 = '{classes[0]}')")
    for key, (mean, norm) in MEANS.items():
        ok = tot = 0
        sc = {c: [] for c in classes}
        for ci, c in enumerate(classes):
            for img in sorted((Path(args.data) / c).glob("*.jpg")):
                v = infer(net, cv2.imread(str(img)), args.imgsz, mean, norm, args.inp, args.outp)
                # out0 ja e probabilidade (Softmax no grafo); usar direto.
                ok += int(np.argmax(v) == ci)
                tot += 1
                sc[c].append(float(v[0]))
        acc = 100.0 * ok / tot if tot else 0.0
        m0, m1 = np.mean(sc[classes[0]]), np.mean(sc[classes[1]])
        print(f"[{key:8}] acc={acc:5.1f}%  P(capacete) medio: "
              f"'{classes[0]}'={m0:.2f}  '{classes[1]}'={m1:.2f}")


if __name__ == "__main__":
    main()
