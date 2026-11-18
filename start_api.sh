#!/usr/bin/env bash
# Laptop 1: start the PathSnap API.   Usage: ./start_api.sh [eval_trips_folder]   (default: data/student_test)
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python)
export PATHSNAP_EVAL_DIR="${1:-data/student_test}"
"$PY" scripts/check_setup.py "$PATHSNAP_EVAL_DIR" || { echo "Fix the failed checks first."; exit 1; }
echo; echo "Starting API on port 8000 (Ctrl+C to stop). Keep this window visible: every request is logged here."; echo
exec "$PY" -m uvicorn api.server:app --host 0.0.0.0 --port 8000
