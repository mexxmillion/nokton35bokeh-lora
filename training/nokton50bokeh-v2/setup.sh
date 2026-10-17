#!/usr/bin/env bash
set -euo pipefail
R=/content/nokton50bokeh-v2
cd /content
[ -d musubi-tuner-v2 ] || git clone https://github.com/kohya-ss/musubi-tuner.git musubi-tuner-v2
cd musubi-tuner-v2
git checkout "$(python -c 'import json;print(json.load(open("/content/nokton50bokeh-v2/versions.json"))["musubi"])')"
pip install -e .
pip install bitsandbytes huggingface_hub
python "$R/download_models.py"
pip freeze > "$R/environment.txt"
