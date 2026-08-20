#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SOURCE_DIR="$ROOT/diagrams"
OUTPUT_DIR="$ROOT/docs/assets/diagrams"
CONFIG="$SOURCE_DIR/mermaid-config.json"
CSS="$SOURCE_DIR/diagram.css"

for command_name in mmdc svgo; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf 'Missing offline renderer: %s\n' "$command_name" >&2
    printf 'Install Mermaid CLI 11 and SVGO outside production dependencies, then retry.\n' >&2
    exit 1
  fi
done

MMD_VERSION=$(mmdc --version | head -n 1)
SVGO_VERSION=$(svgo --version | head -n 1)

mkdir -p "$OUTPUT_DIR"
for source in "$SOURCE_DIR"/*.mmd; do
  name=$(basename "$source" .mmd)
  printf 'render %s\n' "$name"
  mmdc -q -i "$source" -o "$OUTPUT_DIR/$name.svg" -c "$CONFIG" -C "$CSS" -b transparent -w 1600
  svgo --multipass -i "$OUTPUT_DIR/$name.svg" -o "$OUTPUT_DIR/$name.svg" >/dev/null
done

node "$ROOT/scripts/update-diagram-manifest.mjs" "$MMD_VERSION" "$SVGO_VERSION"
node "$ROOT/scripts/check-site.mjs"
