#!/usr/bin/env bash
# Package prismatix as a macOS .app bundle and (optionally) install it.
#
#   bash scripts/install_app.sh            # builds build/Prismatix.app
#   bash scripts/install_app.sh --install  # also copies to /Applications
#
# The bundle's launcher execs THIS repo's venv python. That means the .app is
# a launcher for your checked-out project (honest local packaging), not a
# frozen redistribution; PyInstaller freezing a libre MLX app is future work.

set -euo pipefail
cd "$(dirname "$0")/.."

REPO="$(pwd)"
APP="build/Prismatix.app"
PLIST="$APP/Contents/Info.plist"

echo "1/4  app icon"
bash -c "$(dirname "$0")/../scripts/make_icon.sh" >/dev/null

echo "2/4  bundle skeleton"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

echo "3/4  Info.plist + launcher"
cat > "$PLIST" <<'PLISTEOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Prismatix</string>
  <key>CFBundleDisplayName</key><string>Prismatix — local copilot</string>
  <key>CFBundleIdentifier</key><string>io.github.sentin3l-mrb.prismatix</string>
  <key>CFBundleVersion</key><string>0.1.0</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>prismatix</string>
  <key>CFBundleIconFile</key><string>prismatix</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLISTEOF


# generate the launcher with exact baked-in absolute paths (python repr-quoting
# handles the spaces in this repo path; a bash heredoc mangles them)
.venv/bin/python - <<'PYGEN'
from pathlib import Path
repo = Path.cwd()
launcher = f"""#!/bin/bash
# Prismatix launcher: exec the project's venv python (native window).
REPO={str(repo)!r}
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || {{ echo "venv missing: $PY"; exit 1; }}
cd "$REPO"
exec "$PY" -m prismatix --web
"""
Path("build/Prismatix.app/Contents/MacOS/prismatix").write_text(launcher)
PYGEN
chmod +x "$APP/Contents/MacOS/prismatix"

echo "4/4  icon"
cp assets/prismatix.icns "$APP/Contents/Resources/prismatix.icns"
plutil -lint "$PLIST"

if [[ "${1:-}" == "--install" ]]; then
  rm -rf /Applications/Prismatix.app
  cp -R "$APP" /Applications/Prismatix.app
  echo "installed: /Applications/Prismatix.app"
else
  echo "built:     $REPO/$APP   (add --install to copy to /Applications)"
fi
