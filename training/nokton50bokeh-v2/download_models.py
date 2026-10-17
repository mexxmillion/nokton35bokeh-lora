from pathlib import Path
import json
from huggingface_hub import hf_hub_download
r=Path('/content/nokton50bokeh-v2');versions=json.loads((r/'versions.json').read_text())
files={'Comfy-Org/MiniMax-H3':['diffusion_models/minimax_h3_ref2va_bf16.safetensors','text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors','vae/minimax_h3_video_vae_fp16.safetensors','vae/minimax_h3_audio_vae_fp32.safetensors'],'ostris/minimax_h3_training_adapter':['minimax_h3_ref2va_training_adapter_v1.safetensors']}
for repo,names in files.items():
 for name in names:print(hf_hub_download(repo,name,revision=versions[repo],local_dir=r/'models'),flush=True)
