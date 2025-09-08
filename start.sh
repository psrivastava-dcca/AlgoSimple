#!/bin/bash
set -euo pipefail

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/trading-dashboard"
FRONTEND_DIR="$SCRIPT_DIR/trading-dashboard/trading-frontend"

# Configurable ports (override by exporting before running)
BACKEND_PORT="${PORT:-5050}"
FRONTEND_PORT="${FRONTEND_PORT:-4200}"

# PIDs
BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  echo "\nShutting down..."
  if [[ -n "$FRONTEND_PID" ]] && ps -p "$FRONTEND_PID" >/dev/null 2>&1; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
  if [[ -n "$BACKEND_PID" ]] && ps -p "$BACKEND_PID" >/dev/null 2>&1; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting backend (Flask) on port $BACKEND_PORT..."
(
  cd "$BACKEND_DIR"
  PORT="$BACKEND_PORT" python3 "$BACKEND_DIR/app.py"
) &
BACKEND_PID=$!

echo "Starting frontend (Angular) on port $FRONTEND_PORT..."
(
  cd "$FRONTEND_DIR"
  if [[ ! -d node_modules ]]; then
    echo "node_modules not found; installing dependencies..."
    npm install
  fi
  npx ng serve --port "$FRONTEND_PORT" --open
) &
FRONTEND_PID=$!

echo "\nBackend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "\nAPI:       http://localhost:$BACKEND_PORT"
echo "Frontend:  http://localhost:$FRONTEND_PORT"

# Wait for either to exit
wait -n "$BACKEND_PID" "$FRONTEND_PID" || true

echo "One of the processes exited. See logs above."

