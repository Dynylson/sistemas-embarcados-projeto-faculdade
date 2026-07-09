# Relatório — Detecção de EPI (capacete + colete) no Raspberry Pi 4 com YOLOv8n NCNN

> Data: 2026-06-09 · Projeto EPI · Detecção em tempo real na CPU ARM do Pi 4

## 1. Objetivo

Configurar um Raspberry Pi 4 para rodar **YOLOv8n no formato NCNN** e detectar EPIs —
especificamente **capacete** e **colete** — em tempo real, **somente em CPU** (sem GPU),
com `imgsz=320` para desempenho.

---

## 2. Hardware e ambiente

| Item | Valor |
|---|---|
| Placa | Raspberry Pi 4 Model B Rev 1.5 |
| CPU | ARM Cortex-A72 (ARMv8-A), 4 núcleos, aarch64 |
| RAM | 8 GB |
| SO | Raspberry Pi OS **Lite 64-bit** (Debian 13 "Trixie") |
| Python | 3.13.5 (system, *externally-managed* / PEP 668) |
| Disco | microSD 58 GB (~52 GB livres) |

---

## 3. Acesso ao dispositivo (estado real)

O `CLAUDE.md` indicava `pi@raspberrypi.local`, mas a realidade do dispositivo é diferente:

- **Host:** conectar por **IP** — `10.0.0.165` (o mDNS `raspberrypi.local` **não resolve** nesta rede/Windows).
- **Usuário:** `projeto-embarcados` (definido no Raspberry Pi Imager; **não** é `pi`).
- **Chave:** `~/.ssh/id_raspberry` (ED25519, sem passphrase). Autenticação **só por chave** (senha SSH desabilitada).
- **sudo:** senha definida na configuração do Imager (**não versionada** — manter fora do repositório).

```bash
ssh -i ~/.ssh/id_raspberry -o IdentitiesOnly=yes projeto-embarcados@10.0.0.165
```

> Diagnóstico inicial: o Pi não aparecia por mDNS nem em varredura de porta 22; foi necessário o IP
> direto. A chave precisou ser casada ao usuário correto (`projeto-embarcados`).

---

## 4. Dependências instaladas

### 4.1 Sistema (apt)
```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3.13-venv libgl1 libglib2.0-0 libgomp1
```
(`libgl1`/`libglib2.0-0` para o OpenCV; `libgomp1` para OpenMP/threads.)

### 4.2 Python (venv em `~/epi/venv`)
```bash
python3 -m venv ~/epi/venv && source ~/epi/venv/bin/activate
pip install --upgrade pip
pip install ultralytics ncnn      # puxou torch 2.12, ncnn 1.0.x, opencv 4.13, numpy 2.4
```
Para a **inferência** só são realmente usados: **`ncnn` + `numpy` + `opencv` + `pyyaml`**
(o `torch`/`ultralytics` ficaram instalados mas não são necessários — ver seção 5).

---

## 5. Problema crítico: PyTorch não roda neste Pi (SIGILL)

Ao tentar exportar o modelo **no próprio Pi**, qualquer convolução do PyTorch estourava
`Illegal instruction` (SIGILL):

```
import torch; (torch.randn(4)+1).sum()          # OK
torch.nn.Conv2d(3,16,3)(torch.randn(1,3,32,32)) # Illegal instruction (SIGILL)
```

**Causa:** os wheels de `torch` para **aarch64 / Python 3.13** (tanto `+cu130` quanto `+cpu`) são
compilados com a ARM Compute Library exigindo **ARMv8.2+** (dotprod/i8mm), mas o **Cortex-A72 é
ARMv8-A** e não tem essas instruções.

**Contornos testados que NÃO resolveram:**
- `ATEN_CPU_CAPABILITY=default`
- `torch.backends.mkldnn.enabled = False`
- piwheels (não tem `torch` para cp313)

**Confirmado:** o **ncnn funciona perfeitamente** no A72. Logo, a solução é não depender do torch no Pi.

---

## 6. Arquitetura adotada (correta para produção em Pi)

```
   [ PC com torch (x86) ]                         [ Raspberry Pi 4 ]
   yolov8/EPI .pt  ──export──►  modelo NCNN  ──►  inferência torch-free
   (ultralytics + torch CPU)    (.param/.bin)      (ncnn + numpy + opencv)
```

- **Export `.pt` → NCNN** feito **uma vez no PC** (Windows, venv descartável com torch CPU).
- **Inferência no Pi 100% sem torch**: só `ncnn` (engine) + `numpy`/`opencv` (pré/pós-processo).
- Vantagem: o Pi fica leve e estável; quando houver um modelo de EPI próprio treinado, basta
  re-exportar no PC e copiar os arquivos `.ncnn`.

### Detalhes do pré/pós-processamento (replicam o Ultralytics)
- **Entrada** (`in0`): letterbox para 320×320 (padding 114), BGR→RGB, normalização `/255`.
- **Saída** (`out0`): tensor `(4+nc, 2100)` — 4 box `cx,cy,w,h` (já em px) + `nc` classes (já com sigmoid).
- **Decode:** anchor-free/DFL já embutido no export; aplica-se limiar de confiança + **NMS por classe**
  (`cv2.dnn.NMSBoxesBatched`) e *unmap* do letterbox para coordenadas originais.

---

## 7. Modelo de EPI

**Escolhido:** [`Hansung-Cho/yolov8-ppe-detection`](https://huggingface.co/Hansung-Cho/yolov8-ppe-detection)
(Hugging Face) — **YOLOv8n**, licença **MIT**, treinado no dataset Roboflow Construction Safety.

Classes (10): `0 Hardhat`, `1 Mask`, `2 NO-Hardhat`, `3 NO-Mask`, `4 NO-Safety Vest`,
`5 Person`, `6 Safety Cone`, `7 Safety Vest`, `8 machinery`, `9 vehicle`.

> Como o projeto quer **só capacete e colete**, a saída é **filtrada** para os índices
> **`0` (Hardhat → capacete)** e **`7` (Safety Vest → colete)**.

Export realizado no PC:
```python
from ultralytics import YOLO
YOLO("ppe.pt").export(format="ncnn", imgsz=320)   # gera ppe_ncnn_model/
```

**Alternativa avaliada** (mais precisa, mas licença não declarada):
`hafizqaim/Workspace-Safety-Detection-using-YOLOv8` (YOLOv8n, colete 93.5% / capacete 86.6% mAP50).

---

## 8. Resultados (validação)

Inferência NCNN, `imgsz=320`, 4 threads, CPU do Pi 4:

| Métrica | Valor |
|---|---|
| Tempo de inferência (rede) | **~95 ms** (mediana) |
| Throughput | **~10.5 FPS** |
| Capacete (close-up de trabalhador) | detectado com **0.92** |
| Colete (foto de colete hi-vis nítido) | detectado com **0.91** |

Sanidade do pipeline (modelo COCO yolov8n genérico, `bus.jpg`): ~102 ms, 3 pessoas + 1 ônibus corretos.

### ⚠️ Limitação relevante — objeto pequeno
A `imgsz=320`, EPI **distante/pequeno não é detectado** (numa foto de obra aberta 2254×2818 o
modelo via só "Person"/"NO-Hardhat"). Em tempo real, **enquadrar o trabalhador razoavelmente
próximo** ou aumentar o `imgsz` (custa FPS). Com o EPI grande no quadro, a confiança é alta (0.9+).

---

## 9. Estrutura de arquivos no Pi (`~/epi/`)

```
~/epi/
├── venv/                     # ambiente Python (ncnn/numpy/opencv usados; torch presente mas inutilizado)
├── ppe_ncnn_model/           # modelo de EPI (capacete+colete) em NCNN
│   ├── model.ncnn.param
│   ├── model.ncnn.bin
│   └── metadata.yaml
├── infer_ppe.py              # >>> detector de capacete + colete (script principal)
├── yolov8n_ncnn_model/       # modelo COCO genérico (teste de sanidade)
├── infer_ncnn.py             # detector COCO genérico (teste)
└── out_ppe.jpg               # última imagem anotada gerada
```

---

## 10. Como rodar

```bash
ssh -i ~/.ssh/id_raspberry -o IdentitiesOnly=yes projeto-embarcados@10.0.0.165
cd ~/epi && source venv/bin/activate

# detecção de EPI (capacete + colete) — aceita URL ou caminho local de imagem
python infer_ppe.py "https://commons.wikimedia.org/wiki/Special:FilePath/Highvis.jpg"
python infer_ppe.py /caminho/para/sua/foto.jpg
```
Saída: tempo de inferência, contagem de capacete/colete com confiança e bbox, e
`out_ppe.jpg` anotada (capacete = laranja, colete = verde).

Parâmetros ajustáveis no topo de `infer_ppe.py`: `IMGSZ`, `CONF_THRES` (0.35), `IOU_THRES` (0.45),
`THREADS` (4), `TARGET` (classes filtradas).

---

## 11. Próximos passos sugeridos

1. **Loop em tempo real com câmera** (USB webcam ou módulo PiCamera): capturar frames, inferir por
   frame, desenhar capacete/colete ao vivo com FPS.
2. **Alerta de não-conformidade**: usar as classes `NO-Hardhat`/`NO-Safety Vest` (já existentes no
   modelo) para sinalizar trabalhador sem EPI.
3. **Modelo de EPI próprio**: treinar/afinar no PC, exportar NCNN igual à seção 6 e só trocar os
   arquivos em `~/epi/ppe_ncnn_model/`.
4. **Enxugar o venv** (opcional): remover `torch`/`ultralytics`/`nvidia-*` (~1.5 GB inúteis), já que
   a inferência é torch-free.

---

## 12. Referências
- Modelo EPI: https://huggingface.co/Hansung-Cho/yolov8-ppe-detection
- Ultralytics YOLOv8 (export NCNN): https://docs.ultralytics.com
- ncnn (Tencent): https://github.com/Tencent/ncnn
- Imagens de teste: Wikimedia Commons (`Hard_Hat_Worker_HHW01.JPG`, `Highvis.jpg`) via `Special:FilePath`
