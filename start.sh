#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

mkdir -p logs

echo "=================================================="
echo " Starting Restaurant Operations AI System"
echo "=================================================="

# Check and free ports if occupied
for port in 6333 8000 3000; do
  pid=$(lsof -ti :$port 2>/dev/null || true)
  if [ -n "$pid" ]; then
    echo "Port $port is in use by PID $pid. Releasing..."
    kill -9 $pid 2>/dev/null || true
    sleep 1
  fi
done

# 1. Start Qdrant Vector DB
echo "1/3 Starting Qdrant Vector DB (port 6333)..."
nohup ./backend/bin/qdrant --config-path backend/config/qdrant.yaml < /dev/null > logs/qdrant.log 2>&1 &
echo $! > logs/qdrant.pid
sleep 2

# 2. Start FastAPI Backend
echo "2/3 Starting FastAPI Backend (port 8000)..."
nohup env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./venv/bin/python3 -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 < /dev/null > logs/backend.log 2>&1 &
echo $! > logs/backend.pid

echo "Waiting for backend neural models to initialize..."
for i in {1..30}; do
  if curl -s http://127.0.0.1:8000/api/health | grep -q "healthy"; then
    echo "Backend is ready and warm!"
    break
  fi
  sleep 1
done

# 3. Start React Frontend
echo "3/3 Starting React Frontend (port 3000)..."
if [ ! -d "frontend/dist" ]; then
  npm --prefix frontend run build
fi
nohup ./venv/bin/python3 -m http.server 3000 --directory frontend/dist > logs/frontend.log 2>&1 &
echo $! > logs/frontend.pid
sleep 1

echo "=================================================="
echo "All services successfully started!"
echo "   - Web Interface:    http://127.0.0.1:3000"
echo "   - FastAPI Backend:  http://127.0.0.1:8000"
echo "   - Qdrant Vector DB: http://127.0.0.1:6333"
echo ""
echo "Open in your browser:  http://127.0.0.1:3000"
echo "To stop everything:    ./stop.sh"
echo "To view backend logs:  tail -f logs/backend.log"
echo "=================================================="
