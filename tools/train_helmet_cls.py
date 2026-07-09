#!/usr/bin/env python3
"""Treina o classificador do capacete (YOLOv8n-cls) NO PC e exporta NCNN.

Entrada:  data/helmet_cls_raw/{capacete_ok,nao}/*.jpg  (recortes da cabeca vindos do Pi)
Faz split train/val, treina, e exporta NCNN (imgsz=224). O *_ncnn_model gerado
vai pro Pi (~/epi/helmet_cls_ncnn_model), onde a inferencia e torch-free.

IMPORTANTE (arquitetura do projeto): treinar/exportar SO no PC. No Pi, torch nao roda.

Requer no PC:  pip install ultralytics   (baixa torch; GPU e opcional).

Uso:
  python tools/train_helmet_cls.py --epochs 40
"""
import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def split_train_val(raw, out, val_frac, seed=0):
    random.seed(seed)
    classes = sorted(p.name for p in raw.iterdir() if p.is_dir())
    if not classes:
        raise SystemExit(f"ERRO: sem subpastas de classe em {raw}")
    for cls in classes:
        imgs = sorted((raw / cls).glob("*.jpg"))
        random.shuffle(imgs)
        nval = max(1, int(len(imgs) * val_frac))
        for i, img in enumerate(imgs):
            sub = "val" if i < nval else "train"
            dst = out / sub / cls
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dst / img.name)
        print(f"  {cls}: {len(imgs)} imgs ({nval} val)")
    print(f"[split] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "helmet_cls_raw"))
    ap.add_argument("--out", default=str(ROOT / "data" / "helmet_cls"))
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--model", default="yolov8n-cls.pt")
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    if not raw.exists():
        raise SystemExit(f"ERRO: dataset nao encontrado em {raw}")
    if out.exists():
        shutil.rmtree(out)
    split_train_val(raw, out, args.val_frac)

    from ultralytics import YOLO
    model = YOLO(args.model)
    res = model.train(data=str(out), epochs=args.epochs, imgsz=args.imgsz,
                      batch=32, patience=15, name="helmet_cls")
    best = Path(res.save_dir) / "weights" / "best.pt"
    print(f"[treino] melhor peso: {best}")

    exported = YOLO(str(best)).export(format="ncnn", imgsz=args.imgsz)
    print(f"[export] NCNN em: {exported}")
    print("\nProximo passo — enviar ao Pi:")
    print(f'  scp -i ~/.ssh/id_raspberry -r "{exported}" '
          f'projeto-embarcados@10.0.0.165:~/epi/helmet_cls_ncnn_model')


if __name__ == "__main__":
    main()
