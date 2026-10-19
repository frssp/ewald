#!/bin/bash
# Pack dist/EWALD.app (from packaging/macos/ewald.spec) into a compressed disk image with an
# Applications link: dist/ewald-<version>-macos-<arch>.dmg. macOS only (hdiutil).
set -euo pipefail
cd "$(dirname "$0")/../.."
version=$(PYTHONPATH=src python -c "import ewald; print(ewald.__version__)")
arch=$(uname -m)
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
cp -R dist/EWALD.app "$stage/"
ln -s /Applications "$stage/Applications"
out="dist/ewald-${version}-macos-${arch}.dmg"
rm -f "$out"
hdiutil create -volname "EWALD ${version}" -srcfolder "$stage" -ov -format UDZO "$out"
echo "$out"
