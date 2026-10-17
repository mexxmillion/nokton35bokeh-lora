# Teaching an AI Video Model My Favorite Lens

An article about `nokton35bokeh`, a MiniMax H3 Ref2VA LoRA for the Voigtländer Nokton 35mm look: the dataset approach, Colab A100 training with Musubi Tuner, and before/after tests across eight lighting setups.

Only AI-generated character renders are published. The training photos are personal and are not included.

## GitHub Pages

Pages publishes the root of `gh-pages`. The authoring files live in `site/` on `main`. After an update, commit and push `main`, then publish with:

    git subtree push --prefix site origin gh-pages

Media stays below 25 MiB per file.

## Local preview

    python -m http.server 8766 --directory site

## Reusable training recipe

The working 35mm settings, prepared 1500-step 50mm recipe, private dataset format, backup procedure, and evaluation checklist are saved in [training/README.md](training/README.md). Personal photos, captions describing them, model weights, credentials, and runtime logs are excluded.
