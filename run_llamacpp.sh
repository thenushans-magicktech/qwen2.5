#!/usr/bin/env bash
# Serve a GGUF model with llama.cpp for the "Local model" designer - for models Ollama cannot run yet
# (e.g. LFM3). The designer then lists it as "llamacpp:<alias>".
#
#   ./run_llamacpp.sh <model.gguf | ollama-model-name> [alias] [port]
#   ./run_llamacpp.sh hf.co/saidutta69/LFM3-1.7B-Claude-Fable5.1:Q4_K_M lfm3-fable
#
# then start the designer with:  LLAMACPP_URL=http://127.0.0.1:8095 python3 main.py --serve
#
# CPU only by default (GPU=0): the laptop's integrated GPU crashed (Vulkan "device lost") under load.
# GPU=99 ./run_llamacpp.sh ... offloads all layers to a GPU instead.
#
# Fine-tuned adapter (LoRA, GGUF) on top of the base model - e.g. the MagickHome fine-tune:
#   LORA=finetune/models/magick-gemma-lora-f16.gguf ./run_llamacpp.sh gemma4:E4B magick-gemma 8096
set -euo pipefail
src="${1:?usage: run_llamacpp.sh <model.gguf | ollama-model-name> [alias] [port]}"
port="${3:-8095}"
bin="${LLAMA_SERVER:-$HOME/.unsloth/llama.cpp/llama-server}"
[ -x "$bin" ] || bin="$(command -v llama-server || true)"
[ -n "$bin" ] || { echo "llama-server not found (install llama.cpp or Unsloth Studio, or set LLAMA_SERVER)"; exit 1; }

gguf="$src"
if [ ! -f "$gguf" ]; then
  # an Ollama model name: use the GGUF Ollama already downloaded (no second download)
  name="${src%:*}"; tag="${src##*:}"; [ "$name" = "$src" ] && tag="latest"
  case "$name" in */*) ;; *) name="registry.ollama.ai/library/$name" ;; esac
  manifest="$HOME/.ollama/models/manifests/$name/$tag"
  [ -f "$manifest" ] || { echo "not a file and not an installed Ollama model: $src"; exit 1; }
  digest="$(python3 -c "import json,sys; print(next(l['digest'] for l in json.load(open(sys.argv[1]))['layers'] if l['mediaType'].endswith('.model')))" "$manifest")"
  gguf="$HOME/.ollama/models/blobs/${digest/:/-}"
fi
alias="${2:-$(basename "${src%%:*}")}"
lora_args=()
if [ -n "${LORA:-}" ]; then
  [ -f "$LORA" ] || { echo "LORA file not found: $LORA"; exit 1; }
  lora_args=(--lora "$LORA")
fi
echo "llama.cpp: serving $gguf ${LORA:++ adapter $LORA }as '$alias' on http://127.0.0.1:$port (GPU layers: ${GPU:-0})"
exec "$bin" -m "$gguf" "${lora_args[@]}" --alias "$alias" --host 127.0.0.1 --port "$port" -c "${CTX:-12288}" -ngl "${GPU:-0}" --jinja
