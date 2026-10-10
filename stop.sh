#!/usr/bin/env bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "Stopping Restaurant Operations AI System..."

# Stop processes by PID file if present
if [ -f logs/frontend.pid ]; then
  kill $(cat logs/frontend.pid) 2>/dev/null || true
  rm -f logs/frontend.pid
fi
if [ -f logs/backend.pid ]; then
  kill $(cat logs/backend.pid) 2>/dev/null || true
  rm -f logs/backend.pid
fi
if [ -f logs/qdrant.pid ]; then
  kill $(cat logs/qdrant.pid) 2>/dev/null || true
  rm -f logs/qdrant.pid
fi

# Clean up any lingering processes on ports 3000, 8000, 6333
for port in 3000 8000 6333; do
  pids=$(lsof -ti :$port 2>/dev/null || true)
  if [ -n "$pids" ]; then
    kill -9 $pids 2>/dev/null || true
  fi
done

echo "All services stopped and ports (3000, 8000, 6333) cleared."
