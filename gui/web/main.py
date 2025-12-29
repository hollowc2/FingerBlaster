"""Web UI entry point for FingerBlaster application.

Launches the FastAPI backend server that serves the React frontend.
"""

import logging
import os
import sys

logger = logging.getLogger("FingerBlaster")


def run_web_app(host: str = "0.0.0.0", port: int = 8000):
    """Run the web application (FastAPI backend).
    
    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to listen on (default: 8000)
    """
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
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    FINGER BLASTER - Web API                      ║
╠══════════════════════════════════════════════════════════════════╣
║  API Server:   http://{host}:{port}                              
║  Health Check: http://{host}:{port}/api/health                   
║  WebSocket:    ws://{host}:{port}/ws                             
╠══════════════════════════════════════════════════════════════════╣
║  To run the frontend:                                            ║
║    cd gui/web && npm install && npm run dev                      ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    logger.info(f"Starting Web API server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    """Main entry point for web UI."""
    run_web_app()
