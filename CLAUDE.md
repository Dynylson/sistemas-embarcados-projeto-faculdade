# Projeto EPI - Raspberry Pi 4

## Acesso ao dispositivo
- Host: pi@raspberrypi.local
- Chave SSH: ~/.ssh/id_raspberry

## Objetivo
Rodar YOLOv8n (NCNN) para detecção de EPIs em tempo real.

## Restrições
- Sem GPU, inferência apenas na CPU ARM
- Usar imgsz=320 para melhor performance
- Preferir NCNN sobre ONNX