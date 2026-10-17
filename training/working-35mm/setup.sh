#!/bin/bash
# Runs on the Colab A100 VM: installs musubi-tuner and downloads MiniMax-H3 Ref2VA weights.
set -euo pipefail
cd /content
[ -d musubi-tuner ] || git clone --depth 1 https://github.com/kohya-ss/musubi-tuner.git
cd musubi-tuner && pip install -q -e . && pip install -q "huggingface_hub[cli]" hf_transfer triton
export HF_HUB_ENABLE_HF_TRANSFER=1
M=/content/models; mkdir -p $M
hf download Comfy-Org/MiniMax-H3 \
  diffusion_models/minimax_h3_ref2va_bf16.safetensors \
  text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
  vae/minimax_h3_video_vae_fp16.safetensors \
  vae/minimax_h3_audio_vae_fp32.safetensors --local-dir $M
hf download ostris/minimax_h3_training_adapter minimax_h3_ref2va_training_adapter_v1.safetensors --local-dir $M
df -h /content | tail -1; ls -la $M/*/ $M/*.safetensors
