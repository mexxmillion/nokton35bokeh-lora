#!/usr/bin/env bash
set -euo pipefail
R=/content/nokton50bokeh-v2
# Require mounted persistent storage before spending training compute.
: "${BACKUP_DIR:?Set BACKUP_DIR to a mounted persistent Google Drive directory}"
[[ "$BACKUP_DIR" == /content/drive/MyDrive/* && -d /content/drive/MyDrive ]] || { echo 'Mount Google Drive first'; exit 1; }
mkdir -p "$BACKUP_DIR" "$R/output"
[[ ! -e "$R/TRAINING_STARTED" ]] || { echo 'Run already started; resume explicitly instead of retraining'; exit 1; }
touch "$R/TRAINING_STARTED"
cd /content/musubi-tuner-v2
M=$R/models
python minimax_h3_cache_latents.py --dataset_config "$R/dataset.toml" --task ref2va --one_frame --video_vae "$M/vae/minimax_h3_video_vae_fp16.safetensors" --audio_vae "$M/vae/minimax_h3_audio_vae_fp32.safetensors" --cache_seed 42 --skip_existing
python minimax_h3_cache_text_encoder_outputs.py --dataset_config "$R/dataset.toml" --task ref2va --one_frame --text_encoder "$M/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors" --skip_existing
accelerate launch --num_cpu_threads_per_process 2 --num_processes 1 --mixed_precision bf16 minimax_h3_train_network.py \
 --dataset_config "$R/dataset.toml" --task ref2va --one_frame --video_only \
 --dit "$M/diffusion_models/minimax_h3_ref2va_bf16.safetensors" --convrot_int8 \
 --base_weights "$M/minimax_h3_ref2va_training_adapter_v1.safetensors" \
 --network_module networks.lora_minimax_h3 --network_dim 32 --network_alpha 32 \
 --sdpa --mixed_precision bf16 --gradient_checkpointing --optimizer_type adamw8bit \
 --learning_rate 1e-4 --lr_scheduler constant_with_warmup --lr_warmup_steps 50 \
 --max_train_steps 1500 --save_every_n_steps 250 --save_state --seed 42 \
 --output_dir "$R/output" --output_name nokton50bokeh_v2
