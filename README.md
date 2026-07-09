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
│   ├── live_ppe.py        # detecção AO VIVO via webcam USB + stream MJPEG + áudio + autorização
│   ├── audio_alert.py     # disparo de voz (portaria): autorizado / negado
│   └── helmet_cls.py      # classificador do capacete (YOLOv8-cls NCNN, torch-free)
├── audio/                 # WAV de voz pt-BR (gerados no PC — ver tools/)
│   ├── autorizado.wav
│   └── negado.wav
├── tools/
│   ├── gen_audio_windows.ps1  # gera os WAV de voz via SAPI do Windows (PC)
│   ├── add_cues.py            # embute o jingle antes da voz (PC)
│   ├── collect_dataset.py     # coleta recortes da cabeça p/ treino (Pi)
│   ├── train_helmet_cls.py    # treina YOLOv8-cls e exporta NCNN (PC)
│   └── verify_cls_ncnn.py     # valida o classificador NCNN nos recortes (Pi)
├── deploy/
│   └── epi-live.service   # unit systemd (stream + áudio + autorização no boot)
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

## Áudio — controle de acesso por voz (portaria)

O `live_ppe.py` dispara voz em pt-BR conforme a detecção de **capacete**:

- **capacete presente** → jingle de sucesso (ascendente) + *"Acesso autorizado."*
- **pessoa sem capacete** (classe `NO-Hardhat`) → buzzer grave + *"Acesso negado. Coloque o capacete."*

O efeito sonoro fica **embutido no próprio WAV** antes da voz (um arquivo por evento).

A decisão só vira som após alguns frames estáveis (**debounce**), toca **uma vez por
transição** (não repete com a pessoa parada) e reanuncia só depois que ela sai do quadro.
O áudio roda em thread separada (não trava o stream). Também aparece um banner
**ACESSO AUTORIZADO / NEGADO** no vídeo e a cabeça é marcada em verde (ok) ou vermelho (sem capacete).
A decisão de capacete vem do **classificador** (seção seguinte), não das classes do detector.

**1. Gerar os WAV (no PC Windows):**

```powershell
powershell -ExecutionPolicy Bypass -File tools\gen_audio_windows.ps1  # voz -> audio\voice\
python tools\add_cues.py                                              # jingle+voz -> audio\
```

O `.ps1` usa a voz **Microsoft Maria (pt-BR)** (edite as frases no topo). O `add_cues.py`
sintetiza o efeito sonoro (stdlib pura) e o embute antes da voz nos arquivos finais `audio\*.wav`.

**2. Copiar para o Pi e preparar o áudio:**

```bash
# no PC
scp -i ~/.ssh/id_raspberry -r audio projeto-embarcados@10.0.0.165:~/epi/audio

# no Pi (uma vez): aplay vem no alsa-utils
sudo apt-get install -y alsa-utils
aplay -l                              # listar placas; anote card,device do alto-falante USB
aplay ~/epi/audio/autorizado.wav      # testar (se mudo, veja o device abaixo)
```

**3. Rodar com áudio:**

```bash
EPI_AUDIO_DIR=~/epi/audio python src/live_ppe.py --width 640 --height 480
```

Opções de áudio: `--no-audio` (desliga) · `--audio-dir <pasta>` · `--audio-device plughw:1,0`
(escolhe a placa ALSA — use o `card,device` do `aplay -l`; ou defina `EPI_AUDIO_DEVICE`) ·
`--audio-stable 5` (frames p/ firmar a decisão) · `--audio-interval 2.0` (intervalo mín. entre falas).

> Se `aplay` estiver mudo: confira o volume com `alsamixer` (tecla `F6` p/ trocar de placa) e,
> se o som sair na placa errada, force com `--audio-device plughw:<card>,<device>`.

## Autorização pelo SEU capacete (classificador)

O modelo genérico de EPI **não distingue** um capacete específico de um boné (a classe
`Hardhat` dispara com qualquer coisa na cabeça, e regra de cor falha — capacete e boné
azul-marinho se confundem sob o auto white-balance da webcam). A solução robusta é um
**classificador dedicado** ao seu capacete:

1. o detector acha a `Person` (confiável) → recorta a **cabeça**;
2. um **YOLOv8n-cls** (NCNN, torch-free) classifica o recorte em `capacete_ok` × `nao`;
3. `prob(capacete_ok) ≥ limiar` → autoriza. Boné/cabeça descoberta → nega.

**Pipeline (dados no Pi → treino no PC → NCNN de volta no Pi):**

```bash
# 1) No Pi: coletar recortes (alternando capacete / boné / sem nada), variando pose
EPI_MODEL_DIR=~/epi/ppe_ncnn_model python collect_dataset.py --label capacete_ok --n 100
EPI_MODEL_DIR=~/epi/ppe_ncnn_model python collect_dataset.py --label nao --n 60   # boné
EPI_MODEL_DIR=~/epi/ppe_ncnn_model python collect_dataset.py --label nao --n 40   # descoberto
#    copie ~/epi/dataset para o PC (data/helmet_cls_raw/)

# 2) No PC: treinar + exportar NCNN
pip install ultralytics
python tools/train_helmet_cls.py --epochs 40

# 3) Enviar o modelo ao Pi e validar (escolhe a normalização certa)
scp -i ~/.ssh/id_raspberry -r runs/classify/helmet_cls/weights/best_ncnn_model \
    projeto-embarcados@10.0.0.165:~/epi/helmet_cls_ncnn_model
#    no Pi:
python verify_cls_ncnn.py          # confirma acurácia; normalização calibrada = "plain"
```

O `live_ppe.py` usa `EPI_HELMET_MODEL` (pasta do modelo) e os flags
`--helmet-thresh 0.5` · `--helmet-norm plain`. Sem o modelo, a autorização fica desativada.

> **Escopo:** treinado numa sessão (mesma câmera/luz), separa 100% capacete × boné. Se mudar
> muito a iluminação ou surgir outro tipo de chapéu, recolha mais exemplos e re-treine — o
> pipeline acima é repetível.

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
