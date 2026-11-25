#!/usr/bin/env bash
# Install a fine-tuned Gemma GGUF (downloaded from Colab) into the local Ollama as "magick-gemma".
#
#   bash finetune/install_local.sh ~/Downloads/<model>.gguf [name]
#
# Uses Ollama's built-in Gemma 4 prompt renderer / output parser (as gemma4:E4B does), temperature 0.
# Then run the app with the fine-tuned model and the short prompt:
#   GEMMA_MODEL=magick-gemma GEMMA_USE_KNOWLEDGE=0 MAGICKHOME_AI_PROVIDER=gemma python3 main.py --serve
set -euo pipefail
gguf="${1:?usage: install_local.sh <model.gguf> [name]}"
name="${2:-magick-gemma}"
[ -f "$gguf" ] || { echo "no such file: $gguf"; exit 1; }
gguf="$(cd "$(dirname "$gguf")" && pwd)/$(basename "$gguf")"
mf="$(mktemp)"
cat > "$mf" <<EOF
FROM $gguf
RENDERER gemma4
PARSER gemma4
PARAMETER temperature 0
PARAMETER num_ctx 8192
EOF
echo "Creating Ollama model '$name' from $gguf ($(du -h "$gguf" | cut -f1)) ..."
ollama create "$name" -f "$mf"
rm -f "$mf"
ollama list | grep -E "^$name" || true
echo "Done. Try it: GEMMA_MODEL=$name GEMMA_USE_KNOWLEDGE=0 MAGICKHOME_AI_PROVIDER=gemma python3 main.py --serve"
