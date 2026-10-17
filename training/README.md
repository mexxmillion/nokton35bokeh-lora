# Reproducing the Nokton lens-look training

## What worked, and what is still untested

The **35mm run** produced visible background separation, softer light disks, warmer color and lower contrast in matched reference-to-video comparisons. The original scripts are in `working-35mm/`.

The **50mm v2 run** is prepared for **1500 fresh steps**, using the same training-adapter recipe. It has **not been trained or validated**. Do not describe it as a proven model. Its scripts are in `nokton50bokeh-v2/`.

| Setting | Working 35mm | Prepared 50mm v2 |
|---|---|---|
| Base family | MiniMax H3 Ref2VA | Same |
| Source weights | Full BF16, loaded with `--convrot_int8` | Same |
| Training adapter | Ostris `minimax_h3_ref2va_training_adapter_v1` via `--base_weights` | Same |
| Rank / alpha | 32 / 32 | 32 / 32 |
| Optimizer / learning rate | AdamW8bit / `1e-4` | Same |
| Scheduler | Constant with 50-step warmup | Same |
| Bucket resolution | 1024 × 1024, no upscale | Same |
| Batch / targets | 1 / 40 still images | Same |
| Mode | `--task ref2va --one_frame --video_only` | Same |
| Steps / checkpoint interval | 1000 / 100 | 1500 / 250 |
| Trigger | `nokton35bokeh` | `nokton50bokeh` |
| Resume protection | Weight downloads | Weights + optimizer-state backups to Drive |

The interrupted first 50mm run used guidance loss, lr `5e-5`, alpha16 and 768 buckets. Its step1000 effective weight-update RMS was about **4.29× smaller** than the working 35mm checkpoint, including alpha/rank scaling. This supports a weak-update hypothesis but does not prove which setting caused it, or imply a 4.29× inference multiplier. Loss values across the two objectives are not directly comparable.

## Private dataset preparation

Keep personal photos and cutouts outside Git. Select around40 clear examples with varied subjects, distances, lighting and visible lens character. Preserve original images separately; resize targets to a1536px long edge, honor EXIF orientation and remove watermarks where appropriate. Make subject/object cutouts on light grey as references. Inspect masks manually: poor cutouts can leak background or erase hair, fingers and thin objects.

Each image JSONL record has this shape (replace paths and caption details):

```json
{"image_path":"/content/nokton50bokeh-v2/dataset/targets/example.jpg","caption":"subject_definitions:\n<Subject 1> is the person wearing a blue shirt in <Picture 1>.\n\nsummary:\n[reference generation] The target image is a nokton50bokeh photograph of <Subject 1> seated in a cafe.\n\nretention_analysis:\n<Subject 1> (appears in [Shot 1]): preserved - retain identity and clothing from <Picture 1>.\n\ndetailed_description:\nThe target image is a realistic nokton50bokeh photograph.\n[Shot 1] <Subject 1> sits beside a wooden table, with hanging lamps and other diners in the background.","references":[{"type":"image","path":"/content/nokton50bokeh-v2/dataset/references/example.png"}]}
```

Write factual captions describing the subject, action, framing, background and lighting. Put the trigger in summary and detailed_description. Avoid describing every lens characteristic in every caption: the trigger should carry that association. Reference cutouts from the same target can encourage pose copying; evaluate on unseen subjects and new poses. Tripod-matched sharp/wide-open pairs are a future dataset experiment, not what established these results.

## Colab setup and launch

Authenticate `colab` with the intended Google account, then check `colab sessions` and `colab usage`. Reuse a suitable session if available; otherwise request A100 high memory. Check credits and available storage before launch. Do not include OAuth tokens in this repository.

Create a private upload archive whose root folder is `nokton50bokeh-v2/`, containing the recipe files and your private `dataset/targets`, `dataset/references`, and `dataset/items.jsonl`. Upload and extract to `/content`. The scripts use these exact paths; rename consistently for another run.

```bash
colab new -s nokton50bokeh-v2 --gpu A100 --high-mem
colab upload -s nokton50bokeh-v2 private_training_upload.zip /content/private_training_upload.zip
```

Execute these commands inside Colab (via the notebook or `colab exec`):

```bash
unzip /content/private_training_upload.zip -d /content
bash /content/nokton50bokeh-v2/setup.sh
```

`versions.json` pins Musubi and both Hugging Face model repositories. Setup uses a dedicated `/content/musubi-tuner-v2` checkout and records `pip freeze`; Python packages and CUDA are not fully locked, so validate all CLI help and dependencies on a new runtime.

Mount Google Drive at `/content/drive` before launching (`colab drivemount --help` for the installed CLI's options). Verify `/content/drive/MyDrive` really belongs to the mounted Drive and has enough free capacity for six checkpoints plus six optimizer states. A local directory with that name alone is not persistent storage.

Then execute:

```bash
python /content/nokton50bokeh-v2/launch_remote.py
```

The launcher starts a background training process and a backup worker. Training refuses to launch if Drive is absent or a prior start marker exists. No guidance-loss flags are used with this recipe. The training adapter is merged only for training; **do not add it during inference**.

## Backups, monitoring and recovery

Drive backups go to `/content/drive/MyDrive/nokton50bokeh-v2`. The worker copies checkpoint files and optimizer-state directories after their size/mtime is stable across30-second polls; copies use `.partial` names until complete. Check `backup.log`, markers and destination files. A stable-file check reduces partial-copy risk but does not replace validating a resume checkpoint.

Also download interval checkpoints and optimizer-state directories to your Mac during training. Download logs, configs, dataset manifest and pinned versions. Do not rely on the Mac staying connected: network loss or expired authentication can prevent local monitoring while Colab continues.

Use `colab sessions` and inspect `train.log`/the recorded PID before assuming success or starting another process. A dropped CLI connection does not by itself mean training stopped. A missing runtime does not prove its final checkpoint was backed up.

For recovery with full saved optimizer state, inspect the pinned trainer's `--resume` options and resume the same run toward the original total step limit. If only LoRA weights survived, loading them through `--network_weights` resets the optimizer; label that as a continuation with reset state. Never silently overwrite the original run, restart1500 steps, or count a reset continuation as an exact resume.

## Evaluation

Use the normal Ref2VA base with the compatible Ref2VA turbo adapter and then the trained style LoRA. Start style strength1.0; compare1.5 and2.0 if subtle, checking identity, exposure and artifacts. Do not train with the turbo adapter as a substitute for the training adapter.

Use identical reference, prompt, seed, dimensions, sampler schedule, turbo strength and step count for baseline and style versions. Include an unseen character sheet, daylight, backlighting and night point lights. Keep the trigger present in both baseline and style prompts to isolate the weight effect. Check loader logs to confirm all intended modules attach. For small deltas in Musubi, runtime LoRA attachment avoids BF16 merge rounding.

The requested next comparison uses the user's model reference, baseline/step1000/step1500, roughly0.8MP,8 turbo steps and4s. At24fps deliver96frames; H3 requires generated frame counts `17*n+5`, so generate107 and trim96. Short duration and transfer from still-image training to video remain experimental.

## References

- [Musubi H3 training guide](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/minimax_h3.md)
- [Musubi one-frame guide](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/minimax_h3_1f.md)
- [Ostris training adapter](https://huggingface.co/ostris/minimax_h3_training_adapter)
- [Comfy-Org H3 weights](https://huggingface.co/Comfy-Org/MiniMax-H3)
- [LightX2V turbo adapters](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
