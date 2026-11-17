#!/usr/bin/env bash
set -euo pipefail

COSMOS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TRT_EDGE_ROOT="$COSMOS_ROOT/TensorRT-Edge-LLM"

export EDGELLM_PLUGIN_PATH="$TRT_EDGE_ROOT/build/libNvInfer_edgellm_plugin.so"

exec "$COSMOS_ROOT/.venv/bin/python" -m experimental.server \
  --model "$COSMOS_ROOT/work/Cosmos3-Edge/engines/reasoning" \
  --multimodal-engine-dir "$COSMOS_ROOT/work/Cosmos3-Edge/engines/reasoning" \
  --host 127.0.0.1 \
  --port 8000 \
  --max-input-len 2048 \
  --max-batch-size 1 \
  --max-kv-cache-capacity 4096 \
  --allowed-local-media-path "$COSMOS_ROOT/media"
