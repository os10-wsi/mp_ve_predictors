#!/usr/bin/env bash
# DDMut client
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

echo "=== DDMut client ==="

echo "DDMut is a web service, not distributed code. Nothing to install beyond"
echo "  python -m vepbench install-env ddmut"
echo "Set DDMUT_ENDPOINT if you run it locally; the public server is rate limited."
