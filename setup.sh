#!/usr/bin/env bash
# Setup do projeto EPI no Raspberry Pi: dependencias de sistema + venv + libs Python.
# Idempotente: pode rodar de novo sem problema. NAO inicia o app (ver o fim).
#
#   ./setup.sh
#
set -euo pipefail

cd "$(dirname "$0")"                       # raiz do repo (onde este script esta)

echo "==> [1/3] Dependencias de sistema (apt) ..."
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y python3-venv libgl1 libglib2.0-0 libgomp1 alsa-utils
else
  echo "    (apt-get nao encontrado — pulei; instale manualmente: python3-venv libgl1 libglib2.0-0 libgomp1 alsa-utils)"
fi

echo "==> [2/3] Ambiente virtual Python (venv) ..."
if [ ! -d venv ]; then
  python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

echo "==> [3/3] Dependencias Python (requirements.txt) ..."
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

echo
echo "==> Pronto. Dispositivos de audio disponiveis (para escolher --audio-device):"
aplay -l 2>/dev/null | grep -E "^card" || echo "    (aplay nao listou placas — confira o alto-falante)"

cat <<'EOF'

------------------------------------------------------------------
Como rodar (ative o venv antes: source venv/bin/activate):

  python src/live_ppe.py --width 640 --height 480 --port 8000 --audio-device plughw:0,0

Depois abra em outro PC da rede:  http://<ip-do-pi>:8000/
Descubra o IP com:  hostname -I
Se o som sair na placa errada, troque o plughw:<card>,<device> (veja a lista acima).
------------------------------------------------------------------
EOF
