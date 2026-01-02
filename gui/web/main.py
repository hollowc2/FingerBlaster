"""Web UI entry point for FingerBlaster application.

Launches both the FastAPI backend server and the React frontend dev server.
"""

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("FingerBlaster")

# Global reference to frontend process for cleanup
_frontend_process = None


def _cleanup_frontend():
    """Clean up frontend process on exit."""
    global _frontend_process
    if _frontend_process:
        try:
            _frontend_process.terminate()
            _frontend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _frontend_process.kill()
        except Exception as e:
            logger.warning(f"Error cleaning up frontend process: {e}")
        _frontend_process = None


def _signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info("Received shutdown signal, cleaning up...")
    _cleanup_frontend()
    sys.exit(0)


def run_web_app(host: str = "0.0.0.0", port: int = 8000):
    """Run the web application (both backend and frontend).
    
    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to listen on (default: 8000)
    """
    global _frontend_process
    
    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn not installed. Install with: pip install uvicorn[standard]")
        print("ERROR: uvicorn not installed. Run: pip install uvicorn[standard]")
        sys.exit(1)
    
    try:
        from gui.web.api_server import app
    except ImportError as e:
        logger.error(f"Failed to import api_server: {e}")
        print(f"ERROR: Failed to import api_server: {e}")
        sys.exit(1)
    
    # Get configuration from environment
    host = os.getenv("WEB_API_HOST", host)
    port = int(os.getenv("WEB_API_PORT", port))
    frontend_port = int(os.getenv("WEB_FRONTEND_PORT", 3000))
    
    # Get project root and web directory
    project_root = Path(__file__).parent.parent.parent
    web_dir = project_root / "gui" / "web"
    
    # Check if package.json exists
    package_json = web_dir / "package.json"
    if not package_json.exists():
        logger.error(f"package.json not found at {package_json}")
        print(f"ERROR: package.json not found at {package_json}")
        print("Make sure you're running from the project root.")
        sys.exit(1)
    
    # Check if node_modules exists, if not, try to install
    node_modules = web_dir / "node_modules"
    if not node_modules.exists():
        print("Frontend dependencies not found. Installing...")
        try:
            subprocess.run(
                ["npm", "install"],
                cwd=str(web_dir),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("Frontend dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install frontend dependencies: {e}")
            print("ERROR: Failed to install frontend dependencies.")
            print("Please run manually: cd gui/web && npm install")
            sys.exit(1)
    
    # Register signal handlers for cleanup
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    # Start frontend dev server
    print("Starting frontend dev server...")
    try:
        _frontend_process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=str(web_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        # Give frontend a moment to start
        time.sleep(2)
        
        # Check if process is still running
        if _frontend_process.poll() is not None:
            # Process exited, read output
            output, _ = _frontend_process.communicate()
            logger.error(f"Frontend process exited: {output}")
            print(f"ERROR: Frontend failed to start:\n{output}")
            sys.exit(1)
        
        print("✓ Frontend dev server started")
    except Exception as e:
        logger.error(f"Failed to start frontend: {e}")
        print(f"ERROR: Failed to start frontend: {e}")
        print("Please run manually: cd gui/web && npm run dev")
        sys.exit(1)
    
    # Print startup info
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║              FINGER BLASTER - Web UI (Full Stack)                ║
╠══════════════════════════════════════════════════════════════════╣
║  Frontend:     http://localhost:{frontend_port}                  
║  Backend API:  http://{host}:{port}                              
║  Health Check: http://{host}:{port}/api/health                   
║  WebSocket:    ws://{host}:{port}/ws                             
╠══════════════════════════════════════════════════════════════════╣
║  Open http://localhost:{frontend_port} in your browser          
║  Press Ctrl+C to stop both servers                              
╚══════════════════════════════════════════════════════════════════╝
""")
    
    logger.info(f"Starting Web API server on {host}:{port}")
    
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
        )
    finally:
        # Cleanup on exit
        _cleanup_frontend()


if __name__ == "__main__":
    """Main entry point for web UI."""
    run_web_app()
