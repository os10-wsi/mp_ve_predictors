#!/usr/bin/env bash
# foldseek, which SaProt needs to build its 3Di structural alphabet.
#
# Downloaded from the GitHub release rather than mmseqs.com, because the
# GitHub asset host is reachable on more restricted networks.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

if command -v foldseek >/dev/null 2>&1; then
  echo "foldseek already on PATH: $(command -v foldseek)"
  exit 0
fi

VERSION="${FOLDSEEK_VERSION:-9-427df8a}"
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)  ASSET="foldseek-linux-avx2.tar.gz" ;;
  Linux-aarch64) ASSET="foldseek-linux-arm64.tar.gz" ;;
  Darwin-*)      ASSET="foldseek-osx-universal.tar.gz" ;;
  *) echo "no prebuilt foldseek for $(uname -s)-$(uname -m); build from source" >&2; exit 1 ;;
esac

URL="https://github.com/steineggerlab/foldseek/releases/download/$VERSION/$ASSET"
confirm_download "~50 MB" "foldseek $VERSION ($ASSET)" || exit 1

DEST="$THIRD_PARTY/foldseek"
mkdir -p "$DEST"
curl -fL "$URL" | tar -xz -C "$DEST" --strip-components=1
echo
echo "Installed. Add it to PATH:"
echo "  export PATH=\"$DEST/bin:\$PATH\""
