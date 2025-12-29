#!/bin/bash
#
# FingerBlaster Web UI Launcher
# 
# Starts both the FastAPI backend and Vite frontend dev server.
# Usage: ./scripts/run_web.sh
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Project root (parent of scripts directory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║              FINGER BLASTER - Web UI Launcher                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check for .env file
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}[!]${NC} No .env file found in project root."
    echo -e "${YELLOW}[!]${NC} Copy env.example to .env and configure your settings."
    echo ""
fi

# Check for Python venv
if [ -d "venv" ]; then
    echo -e "${GREEN}[+]${NC} Activating Python virtual environment..."
    source venv/bin/activate
else
    echo -e "${YELLOW}[!]${NC} No venv found. Using system Python."
fi

# Check dependencies
echo -e "${GREEN}[+]${NC} Checking Python dependencies..."
if ! python -c "import uvicorn, fastapi" 2>/dev/null; then
    echo -e "${RED}[!]${NC} Missing Python dependencies. Installing..."
    pip install -r requirements.txt
fi

# Check Node modules
if [ ! -d "gui/web/node_modules" ]; then
    echo -e "${GREEN}[+]${NC} Installing frontend dependencies..."
    cd gui/web
    npm install
    cd "$PROJECT_ROOT"
fi

# Start backend in background
echo -e "${GREEN}[+]${NC} Starting FastAPI backend on port ${WEB_API_PORT:-8000}..."
python main.py --web &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Check if backend started
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}[!]${NC} Backend failed to start!"
    exit 1
fi

# Start frontend
echo -e "${GREEN}[+]${NC} Starting Vite frontend on port 3000..."
cd gui/web
npm run dev &
FRONTEND_PID=$!

cd "$PROJECT_ROOT"

echo ""
echo -e "${GREEN}[+]${NC} Both servers started successfully!"
echo ""
echo -e "${CYAN}  Backend API:   ${NC}http://localhost:${WEB_API_PORT:-8000}"
echo -e "${CYAN}  Frontend:      ${NC}http://localhost:3000"
echo ""
echo -e "${GREEN}✓${NC} Open ${CYAN}http://localhost:3000${NC} in your browser to view the UI"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop both servers${NC}"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}[!]${NC} Shutting down..."
    
    # Stop frontend first (usually quick)
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        echo -e "${GREEN}[+]${NC} Stopping frontend..."
        kill -INT $FRONTEND_PID 2>/dev/null || true
        sleep 1
        # Force kill if still running
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            kill -9 $FRONTEND_PID 2>/dev/null || true
        fi
    fi
    
    # Stop backend with SIGINT (graceful shutdown)
    if kill -0 $BACKEND_PID 2>/dev/null; then
        echo -e "${GREEN}[+]${NC} Stopping backend (graceful)..."
        kill -INT $BACKEND_PID 2>/dev/null || true
        
        # Wait up to 5 seconds for graceful shutdown
        for i in {1..5}; do
            if ! kill -0 $BACKEND_PID 2>/dev/null; then
                break
            fi
            sleep 1
        done
        
        # Force kill if still running
        if kill -0 $BACKEND_PID 2>/dev/null; then
            echo -e "${YELLOW}[!]${NC} Backend not responding, force killing..."
            kill -9 $BACKEND_PID 2>/dev/null || true
        fi
    fi
    
    echo -e "${GREEN}[+]${NC} Shutdown complete."
    exit 0
}

# Trap Ctrl+C
trap cleanup INT TERM

# Wait for either process to exit
wait $BACKEND_PID $FRONTEND_PID

