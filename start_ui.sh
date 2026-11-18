#!/usr/bin/env bash
# Laptop 2: serve the UI.   Usage: ./start_ui.sh <laptop1_ip>
cd "$(dirname "$0")/ui"
PY=$(command -v python3 || command -v python)
IP="${1:-localhost}"
echo "Open this in Chrome:  http://localhost:8080/?api=http://$IP:8000"
echo "(Ctrl+C to stop)"
exec "$PY" -m http.server 8080
