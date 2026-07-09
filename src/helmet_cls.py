#!/usr/bin/env python3
"""Classificador do capacete (YOLOv8n-cls) via NCNN, torch-free (roda no Pi).

Recebe o recorte da cabeca (BGR) e devolve a probabilidade de 'capacete_ok'.
Classes na ordem alfabetica das pastas de treino: 0=capacete_ok, 1=nao.

A normalizacao ('imagenet' vs 'plain') deve casar com o treino do YOLOv8-cls;
confirme com tools/verify_cls_ncnn.py (no Pi) antes de integrar ao live.
"""
from pathlib import Path

import cv2
import numpy as np
import ncnn

# ncnn: out = (x - mean) * norm, com x em 0..255 e canais em RGB.
_MEAN = {
    "imagenet": [0.485 * 255, 0.456 * 255, 0.406 * 255],
    "plain": [0.0, 0.0, 0.0],
}
_NORM = {
    "imagenet": [1 / (0.229 * 255), 1 / (0.224 * 255), 1 / (0.225 * 255)],
    "plain": [1 / 255.0] * 3,
}


def head_crop(frame, pbox, size=224):
    """Recorte QUADRADO da cabeca a partir do box da pessoa (a,b,x,y).
    DEVE ser identico ao usado na coleta (tools/collect_dataset.py)."""
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


class HelmetClassifier:
    def __init__(self, model_dir, imgsz=224, threads=2, norm="plain",
                 input_name="in0", output_name="out0"):
        d = Path(model_dir)
        self.imgsz = imgsz
        self.input_name = input_name
        self.output_name = output_name
        self.mean, self.norm = _MEAN[norm], _NORM[norm]
        self.net = ncnn.Net()
        self.net.opt.num_threads = threads
        self.net.opt.use_vulkan_compute = False
        self.net.load_param(str(d / "model.ncnn.param"))
        self.net.load_model(str(d / "model.ncnn.bin"))

    def prob_capacete(self, crop_bgr):
        """Probabilidade [0..1] de o recorte ser 'capacete_ok'."""
        img = cv2.resize(crop_bgr, (self.imgsz, self.imgsz))
        mat = ncnn.Mat.from_pixels(np.ascontiguousarray(img),
                                   ncnn.Mat.PixelType.PIXEL_BGR2RGB, self.imgsz, self.imgsz)
        mat.substract_mean_normalize(self.mean, self.norm)
        ex = self.net.create_extractor()
        ex.input(self.input_name, mat)
        _, out = ex.extract(self.output_name)
        v = np.array(out).flatten().astype(np.float32)
        # o grafo NCNN ja termina em Softmax -> out0 e probabilidade (nao re-aplicar).
        return float(v[0]) if v.size else 0.0   # indice 0 = capacete_ok
