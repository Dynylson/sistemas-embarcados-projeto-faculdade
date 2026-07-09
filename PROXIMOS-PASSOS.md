# Próximos Passos — Detecção de EPI no Raspberry Pi

> Roadmap do projeto. Veja [README.md](README.md) (visão geral) e [RELATORIO.md](RELATORIO.md) (histórico/decisões).
> Atualizado: 2026-06-09.

## ✅ Estado atual (já concluído)

- [x] Pi 4 acessível por SSH (`projeto-embarcados@10.0.0.165`, chave `id_raspberry`).
- [x] Ambiente configurado (venv torch-free: `ncnn` + `numpy` + `opencv`).
- [x] Arquitetura definida: export NCNN no PC, inferência torch-free no Pi (torch não roda no A72).
- [x] Modelo de EPI implantado (Hansung-Cho YOLOv8n, MIT) filtrando **capacete (0)** e **colete (7)**.
- [x] Inferência validada: ~95 ms (~10.5 FPS) @ imgsz=320; capacete 0.92 / colete 0.91.
- [x] Webcam USB (Logitech C270) reconhecida em `/dev/video0`.
- [x] **Detecção AO VIVO** com stream MJPEG em `http://10.0.0.165:8000/`.
- [x] Código versionável no repo (`src/`, `models/`, `requirements.txt`, `.gitignore`).
- [x] **Áudio de controle de acesso** (portaria): jingle + voz pt-BR — *"Acesso autorizado"* /
  *"Acesso negado. Coloque o capacete."* no jack P2 (`plughw:0,0`), com debounce/borda.
- [x] **Autostart via systemd** (`epi-live.service`, enabled) — stream + áudio sobem no boot.
- [x] **Autorização pelo capacete específico** (classificador YOLOv8-cls NCNN, torch-free):
  detecta `Person` → recorta cabeça → classifica `capacete_ok`×`nao`. Resolve o boné passando
  como capacete (regra de cor falhou: capacete e boné azul-marinho + auto-WB da C270).
  100% na validação; ~6 FPS ao vivo (2 inferências/frame). `norm=plain`, limiar 0.5.

---

## 🔜 Próximos passos

### P1 — Curto prazo (fazer primeiro)

#### 1. Calibração e validação em campo
- **Objetivo:** garantir que detecta capacete/colete reais com poucos falsos positivos/negativos.
- **Passos:** testar com capacete e colete de verdade na frente da webcam, em distâncias/iluminações
  variadas; ajustar `--conf` (começar em 0.35; subir se houver falso positivo, descer se perder detecção);
  conferir se o colete usado (modelo brasileiro) é reconhecido bem.
- **Pronto quando:** detecção estável a ~1–3 m da câmera com confiança consistente > 0.5.
- **Nota:** a `imgsz=320` o EPI precisa estar razoavelmente grande no quadro (ver item 6).

#### 2. Autostart no boot (serviço systemd)
- **Objetivo:** o stream sobe sozinho quando o Pi liga, reinicia se cair.
- **Passos:** criar `/etc/systemd/system/epi-live.service`:
  ```ini
  [Unit]
  Description=EPI live detection (capacete+colete)
  After=network-online.target

  [Service]
  User=projeto-embarcados
  SupplementaryGroups=audio
  WorkingDirectory=/home/projeto-embarcados/epi
  Environment=EPI_MODEL_DIR=/home/projeto-embarcados/epi/ppe_ncnn_model
  Environment=EPI_AUDIO_DIR=/home/projeto-embarcados/epi/audio
  ExecStart=/home/projeto-embarcados/epi/venv/bin/python /home/projeto-embarcados/epi/live_ppe.py --width 640 --height 480 --port 8000
  Restart=on-failure
  RestartSec=3

  [Install]
  WantedBy=multi-user.target
  ```
  ```bash
  sudo systemctl daemon-reload
  sudo systemctl enable --now epi-live.service
  sudo systemctl status epi-live.service
  ```
- **Pronto quando:** reiniciar o Pi e o `http://10.0.0.165:8000/` voltar sozinho.

#### 3. Alerta de não-conformidade (sem EPI)
- **Objetivo:** sinalizar trabalhador **sem** capacete/colete (segurança é o foco do projeto).
- **Passos:** incluir as classes que o modelo já tem — `2 NO-Hardhat`, `4 NO-Safety Vest` — desenhando-as
  em **vermelho** com rótulo "SEM CAPACETE/COLETE"; opcional: status global "CONFORME / NÃO CONFORME" no overlay.
- **Pronto quando:** pessoa sem capacete aparece marcada em vermelho ao vivo.
- **Nota:** ajustar o dicionário `TARGET` em `src/infer_ppe.py` para incluir 2 e 4.

### P2 — Médio prazo

#### 4. Registro de eventos / evidências
- **Objetivo:** guardar histórico de violações.
- **Passos:** ao detectar "sem EPI", gravar linha em CSV (timestamp, tipo, confiança) e salvar snapshot
  `violacoes/AAAA-MM-DD_HH-MM-SS.jpg`. Evitar spam (1 registro a cada N segundos por evento).
- **Pronto quando:** existe um log consultável + imagens das violações.

#### 5. Robustez / produção
- **Objetivo:** rodar sem babá.
- **Passos:** reconexão automática se a câmera cair (`cap.read()` falhar → reabrir); tratamento de exceções
  no loop; watchdog via `Restart=on-failure` (já no systemd); rotacionar `live.log`.
- **Pronto quando:** desconectar/reconectar a webcam e o serviço se recuperar sozinho.

#### 6. Ajuste de desempenho / alcance
- **Objetivo:** equilibrar FPS x detecção de objetos pequenos/distantes.
- **Opções:** `--imgsz 416/512` (detecta mais longe, FPS cai); processar 1 a cada 2 frames; recortar ROI
  (região de interesse) se a câmera for fixa. Medir FPS real em cada ajuste.
- **Pronto quando:** configuração escolhida atende ao cenário real de uso (distância da câmera).

### P3 — Longo prazo (opcional)

#### 7. Modelo de EPI próprio
- **Quando:** se a acurácia do modelo atual (capacete/colete) não bastar no cenário real.
- **Passos:** coletar/anotar imagens do ambiente real (ou usar dataset Roboflow Construction Safety),
  treinar YOLOv8n **no PC/GPU**, exportar NCNN (`imgsz=320`) **no PC** e só trocar os arquivos em
  `models/ppe_ncnn_model/` (ajustar `TARGET` se os índices mudarem). **Nunca treinar/exportar no Pi.**
- **Alternativa pronta:** `hafizqaim/Workspace-Safety-Detection` (YOLOv8n, colete 93.5% mAP50, licença não declarada).

#### 8. Interface / integração
- Melhorar a página web (contadores, status, múltiplas câmeras) ou expor uma API/MQTT para integrar
  com um painel/sistema de portaria.

#### 9. Versionamento (git) — *pendente*
- `git init` + primeiro commit da estrutura atual e push para um remote (GitHub/GitLab).
- **Pronto quando:** repositório publicado com README/RELATORIO/código (sem a pasta `venv/`).

---

## ⚠️ Decisões / pontos em aberto
- **Limiar de confiança (`--conf`)**: definir após teste em campo (item 1).
- **Resolução de captura vs FPS**: 640×480 (atual, mais fluido) vs 1280×720 (mais detalhe).
- **Colete BR**: confirmar se o "Safety Vest" do dataset cobre os coletes usados aqui; senão → item 7.
- **Visualização headless**: hoje é stream MJPEG; avaliar se basta ou se quer gravar vídeo/painel.

## 🗺️ Sugestão de ordem
**1 (calibrar) → 3 (alerta sem EPI) → 2 (autostart) → 4 (registro) → 5/6 (robustez/perf)**,
deixando 7–9 conforme a necessidade do projeto.
