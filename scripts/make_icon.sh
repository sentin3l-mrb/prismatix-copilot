#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python scripts/make_icon.py
mkdir -p assets/appicon.iconset
for s in 16 32 64 128 256 512; do
  sips -s format png --resampleWidth "$s" assets/appicon.pdf --out "assets/appicon.iconset/icon_${s}x${s}.png" >/dev/null
  if [ "$s" -le 256 ]; then
    sips -s format png --resampleWidth $((s * 2)) assets/appicon.pdf --out "assets/appicon.iconset/icon_${s}x${s}@2x.png" >/dev/null
  fi
done
iconutil -c icns assets/appicon.iconset -o assets/prismatix.icns
