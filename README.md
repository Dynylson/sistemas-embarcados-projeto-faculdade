# EPI Detection — Raspberry Pi 4 + YOLOv8n (NCNN)

Detecção de **EPI (capacete e colete)** em tempo real na **CPU ARM** de um Raspberry Pi 4,
usando **YOLOv8n exportado para NCNN** e inferência **torch-free** (só `ncnn` + `numpy` + `opencv`).

> 📄 Histórico completo do que foi feito, decisões e diagnósticos: veja **[RELATORIO.md](RELATORIO.md)**.

## Por que torch-free?

No Pi 4 (Cortex-A72, Debian 13/Python 3.13) o **PyTorch não roda** — qualquer convolução estoura
`Illegal instruction` (SIGILL), pois os wheels aarch64 exigem ARMv8.2+. O **ncnn funciona normalmente**.
Por isso a arquitetura é:

```
[ PC com torch ]  --export-->  modelo NCNN (.param/.bin)  -->  [ Pi: inferência só com ncnn ]
```

## Estrutura

```
.
├── README.md              # este arquivo
├── RELATORIO.md           # relatório completo / runbook
├── requirements.txt       # deps de inferência (torch-free)
├── .gitignore
├── src/
│   ├── infer_ppe.py       # detector de capacete + colete (imagem única)
│   └── live_ppe.py        # detecção AO VIVO via webcam USB + stream MJPEG
└── models/
    └── ppe_ncnn_model/    # modelo YOLOv8n NCNN (Hansung-Cho, MIT) — capacete=0, colete=7
        ├── model.ncnn.param
        ├── model.ncnn.bin
        └── metadata.yaml
```

## Quickstart (no Raspberry Pi)

```bash
# deps de sistema (uma vez)
sudo apt-get install -y python3-venv libgl1 libglib2.0-0 libgomp1

# ambiente
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# rodar (imagem local, URL, ou exemplo padrão)
python src/infer_ppe.py caminho/para/foto.jpg
python src/infer_ppe.py https://commons.wikimedia.org/wiki/Special:FilePath/Highvis.jpg
```

Saída: tempo de inferência, contagem de **capacete/colete** com confiança e bbox, e uma imagem
anotada `out_ppe.jpg` (capacete = laranja, colete = verde).

Opções: `--imgsz 320 --conf 0.35 --iou 0.45 --threads 4 --runs 20`.
O diretório do modelo pode ser trocado via `EPI_MODEL_DIR=/caminho`.

## Ao vivo (webcam USB + stream MJPEG)

Detecção em tempo real com uma webcam USB (testado: Logitech C270). Como o OS é headless,
o resultado é servido por HTTP — abra no navegador de outra máquina:

```bash
# no Pi (modelo via EPI_MODEL_DIR se o layout não for o do repo)
python src/live_ppe.py --width 640 --height 480 --port 8000
# depois, no PC:  http://<ip-do-pi>:8000/
```

Mostra as caixas de capacete/colete e um overlay com FPS/tempo de inferência.
Opções: `--device 0 --conf 0.4 --threads 4`. Parar: `Ctrl+C` (ou `pkill -f live_ppe.py`).

## Desempenho (medido)

| | |
|---|---|
| Inferência | ~95 ms (imgsz=320, 4 threads, CPU Pi 4) |
| Throughput | ~10.5 FPS |
| Capacete / Colete | 0.92 / 0.91 de confiança em imagens nítidas |

⚠️ A `imgsz=320`, EPI pequeno/distante não é detectado — enquadrar o trabalhador de perto ou aumentar `imgsz`.

## Trocar por um modelo de EPI próprio

Treine/exporte **no PC** (não no Pi) e substitua os arquivos em `models/ppe_ncnn_model/`:

```python
from ultralytics import YOLO
YOLO("seu_modelo.pt").export(format="ncnn", imgsz=320)
```

Ajuste o dicionário `TARGET` em `src/infer_ppe.py` se os índices das classes mudarem.

## Licença / créditos

- Modelo: [Hansung-Cho/yolov8-ppe-detection](https://huggingface.co/Hansung-Cho/yolov8-ppe-detection) (YOLOv8n, MIT)
- YOLOv8/Ultralytics (AGPL-3.0) · [ncnn](https://github.com/Tencent/ncnn) (Tencent)
