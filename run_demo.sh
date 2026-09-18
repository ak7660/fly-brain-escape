#!/usr/bin/env bash
# Start the fly-brain escape demo: serves web/ and opens it in your browser.
#   ./run_demo.sh            normal (you launch threats)
#   ./run_demo.sh demo       threats launch automatically (for recordings)
#   PORT=8001 ./run_demo.sh  use another port if 8000 is busy
# Stop with Ctrl+C.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8000}"
URL="http://localhost:${PORT}/"
[[ "${1:-}" == "demo" ]] && URL="${URL}?demo=1"

if ss -ltn 2>/dev/null | grep -q ":${PORT}\b"; then
  echo "Port ${PORT} is already in use. Try: PORT=8001 $0 ${1:-}"
  exit 1
fi

echo "Serving the fly brain demo at ${URL}  (Ctrl+C to stop)"
( sleep 1; command -v xdg-open >/dev/null && xdg-open "${URL}" >/dev/null 2>&1 || true ) &
exec python3 -m http.server -d web "${PORT}" --bind 127.0.0.1
