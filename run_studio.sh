#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "======================================================="
echo "  Starting Mirror Crystal Live Studio (Port 8787)..."
echo "  Zero-GPU Weightless Cognitive Engine"
echo "======================================================="
echo ""

# Default to process sandbox on developer systems unless container is explicitly requested
export MIRROR_SANDBOX_MODE="${MIRROR_SANDBOX_MODE:-process}"

if [[ -f ".venv/bin/python" ]]; then
    PY_BIN=".venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
    PY_BIN="uv run python"
elif command -v python3 >/dev/null 2>&1; then
    PY_BIN="python3"
else
    echo "Python 3.11+ or uv was not found. Please install Python or uv."
    exit 1
fi

echo "Opening browser at http://127.0.0.1:8787/ ..."
if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://127.0.0.1:8787/" >/dev/null 2>&1 &
elif command -v open >/dev/null 2>&1; then
    open "http://127.0.0.1:8787/" >/dev/null 2>&1 &
fi

exec ${PY_BIN} -m mirror7_backend.http_server
