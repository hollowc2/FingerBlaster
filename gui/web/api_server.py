"""FastAPI backend server for FingerBlaster Web GUI.

Bridges the React frontend with FingerBlasterCore via:
- REST endpoints for actions (order, flatten, cancel, size)
- WebSocket for real-time data streaming (prices, countdown, analytics)

Security features:
- Rate limiting via slowapi
- Optional API key authentication
"""

import asyncio
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from functools import wraps
from typing import List, Optional, Any, Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.core import FingerBlasterCore
from src.analytics import AnalyticsSnapshot, TimerUrgency, EdgeDirection

logger = logging.getLogger("FingerBlaster.WebAPI")
# Ensure our logger shows up
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)


# =============================================================================
# Rate Limiting
# =============================================================================

limiter = Limiter(key_func=get_remote_address)


# =============================================================================
# API Key Authentication
# =============================================================================

# Generate a secure API key on startup if not provided via env
_generated_api_key = secrets.token_urlsafe(32)
API_KEY = os.getenv("FB_API_KEY", "")  # Empty = no auth required
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key() -> str:
    """Get the active API key (generated if not set via env)."""
    return API_KEY if API_KEY else _generated_api_key


async def verify_api_key(request: Request, api_key: Optional[str] = Security(API_KEY_HEADER)) -> bool:
    """Verify API key if authentication is enabled."""
    # Skip auth if API_KEY env var is empty (dev mode)
    if not API_KEY:
        return True
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    if not secrets.compare_digest(api_key, API_KEY):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )
    
    return True


# =============================================================================
# Pydantic Models
# =============================================================================

class OrderRequest(BaseModel):
    """Request to place an order."""
    side: str = Field(..., pattern="^(Up|Down)$", description="Order side: Up or Down")
    size: Optional[float] = Field(None, gt=0, description="Order size in USDC")


class SizeRequest(BaseModel):
    """Request to adjust order size."""
    action: str = Field(..., pattern="^(up|down)$", description="Size action: up or down")


class OrderResponse(BaseModel):
    """Response after order submission."""
    status: str
    side: str
    size: float


class StatusResponse(BaseModel):
    """Generic status response."""
    status: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    connected: bool
    market_active: bool


# =============================================================================
# WebSocket Connection Manager
# =============================================================================

class ConnectionManager:
    """Manages WebSocket connections with thread-safe broadcasting."""
    
    def __init__(self):
        self._clients: List[WebSocket] = []
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self._clients.append(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self._clients)}")
    
    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            if websocket in self._clients:
                self._clients.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(self._clients)}")
    
    async def broadcast(self, event: str, data: dict) -> None:
        """Broadcast event to all connected clients."""
        if not self._clients:
            return
        
        message = json.dumps({"event": event, "data": data})
        
        async with self._lock:
            clients_snapshot = list(self._clients)
        
        disconnected = []
        for client in clients_snapshot:
            try:
                await client.send_text(message)
            except Exception as e:
                logger.debug(f"Failed to send to client: {e}")
                disconnected.append(client)
        
        # Clean up disconnected clients
        if disconnected:
            async with self._lock:
                for client in disconnected:
                    if client in self._clients:
                        self._clients.remove(client)
    
    @property
    def client_count(self) -> int:
        """Return number of connected clients."""
        return len(self._clients)


# =============================================================================
# Global State
# =============================================================================

core: Optional[FingerBlasterCore] = None
manager = ConnectionManager()
background_task: Optional[asyncio.Task] = None


# =============================================================================
# Core Callback Handlers (emit to WebSocket clients)
# =============================================================================

def _schedule_broadcast(event: str, data: dict) -> None:
    """Schedule a broadcast on the event loop."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(manager.broadcast(event, data))
    except RuntimeError:
        # No running loop - skip broadcast
        pass


def on_market_update(strike: str, ends: str) -> None:
    """Handle market update from core."""
    _schedule_broadcast("market_update", {
        "priceToBeat": strike,
        "ends": ends,
    })


def on_btc_price_update(price: float) -> None:
    """Handle BTC price update from core."""
    _schedule_broadcast("btc_price", {"price": price})


def on_price_update(yes_price: float, no_price: float, best_bid: float, best_ask: float) -> None:
    """Handle Up/Down price update from core."""
    yes_spread, no_spread = FingerBlasterCore.calculate_spreads(best_bid, best_ask)
    _schedule_broadcast("price_update", {
        "yesPrice": yes_price,
        "noPrice": no_price,
        "bestBid": best_bid,
        "bestAsk": best_ask,
        "yesSpread": yes_spread,
        "noSpread": no_spread,
    })


def on_account_stats_update(
    balance: float, 
    yes_bal: float, 
    no_bal: float, 
    size: float,
    avg_yes: Optional[float] = None,
    avg_no: Optional[float] = None
) -> None:
    """Handle account stats update from core."""
    _schedule_broadcast("account_stats", {
        "balance": balance,
        "yesBalance": yes_bal,
        "noBalance": no_bal,
        "selectedSize": size,
        "avgEntryYes": avg_yes,
        "avgEntryNo": avg_no,
    })


def on_countdown_update(time_str: str, urgency: Optional[TimerUrgency], seconds_remaining: int) -> None:
    """Handle countdown timer update from core."""
    urgency_str = "normal"
    if urgency == TimerUrgency.CRITICAL:
        urgency_str = "critical"
    elif urgency == TimerUrgency.WATCHFUL:
        urgency_str = "watchful"
    
    _schedule_broadcast("countdown", {
        "timeLeft": time_str,
        "urgency": urgency_str,
        "secondsRemaining": seconds_remaining,
    })


def on_prior_outcomes_update(outcomes: List[str]) -> None:
    """Handle prior outcomes update from core."""
    _schedule_broadcast("prior_outcomes", {"outcomes": outcomes})


def on_resolution(resolution: Optional[str]) -> None:
    """Handle market resolution from core."""
    _schedule_broadcast("resolution", {"resolution": resolution})


def on_log(message: str) -> None:
    """Handle log message from core."""
    _schedule_broadcast("log", {"message": message})


def on_chart_update(*args) -> None:
    """Handle chart data update from core."""
    if len(args) == 3 and args[2] == 'btc':
        # BTC chart: (prices: List[float], price_to_beat: Optional[float], 'btc')
        prices, strike_val, _ = args
        _schedule_broadcast("btc_chart", {
            "prices": list(prices) if prices else [],
            "priceToBeat": strike_val,
        })
    elif len(args) >= 1:
        # Probability chart: (history: List[Tuple[float, float]])
        history = args[0]
        _schedule_broadcast("probability_chart", {
            "data": [{"x": x, "y": y} for x, y in history] if history else [],
        })


def on_analytics_update(snapshot: AnalyticsSnapshot) -> None:
    """Handle analytics snapshot from core."""
    # Convert EdgeDirection enum to string
    def edge_to_str(edge: Optional[EdgeDirection]) -> Optional[str]:
        if edge is None:
            return None
        return edge.value if hasattr(edge, 'value') else str(edge)
    
    _schedule_broadcast("analytics", {
        "basisPoints": snapshot.basis_points,
        "zScore": snapshot.z_score,
        "sigmaLabel": snapshot.sigma_label,
        "fairValueYes": snapshot.fair_value_yes,
        "fairValueNo": snapshot.fair_value_no,
        "edgeYes": edge_to_str(snapshot.edge_yes),
        "edgeNo": edge_to_str(snapshot.edge_no),
        "edgeBpsYes": snapshot.edge_bps_yes,
        "edgeBpsNo": snapshot.edge_bps_no,
        "unrealizedPnl": snapshot.total_unrealized_pnl,
        "pnlPercentage": snapshot.pnl_percentage,
        "yesAskDepth": snapshot.yes_ask_depth,
        "noAskDepth": snapshot.no_ask_depth,
        "estimatedSlippageYes": snapshot.estimated_slippage_yes,
        "estimatedSlippageNo": snapshot.estimated_slippage_no,
        "regimeDirection": snapshot.regime_direction,
        "regimeStrength": snapshot.regime_strength,
        "oracleLagMs": snapshot.oracle_lag_ms,
        "timerUrgency": snapshot.timer_urgency.value if snapshot.timer_urgency else "normal",
    })


# =============================================================================
# Background Update Loop
# =============================================================================

async def run_update_loop() -> None:
    """Run core update tasks in a loop."""
    global core
    if not core:
        return
    
    print("[UPDATE LOOP] Starting...", flush=True)
    
    # Start RTDS for real-time BTC prices
    print("[UPDATE LOOP] Starting RTDS...", flush=True)
    await core.start_rtds()
    print("[UPDATE LOOP] RTDS started", flush=True)
    
    # Immediately try to discover market on startup
    print("[UPDATE LOOP] Initial market discovery...", flush=True)
    try:
        await core.update_market_status()
        print("[UPDATE LOOP] Initial market discovery completed", flush=True)
        logger.info("Initial market discovery completed")
    except Exception as e:
        print(f"[UPDATE LOOP] Initial market discovery failed: {e}", flush=True)
        logger.warning(f"Initial market discovery failed: {e}")
    
    config = core.config
    
    # Track last update times for each task
    last_market_update = 0.0
    last_btc_update = 0.0
    last_stats_update = 0.0
    last_countdown_update = 0.0
    last_analytics_update = 0.0
    
    while True:
        try:
            now = time.time()
            
            # Market status (every 5s) - with timeout protection
            if now - last_market_update >= config.market_status_interval:
                try:
                    await asyncio.wait_for(core.update_market_status(), timeout=12.0)
                except asyncio.TimeoutError:
                    logger.warning("update_market_status timed out")
                except Exception as e:
                    logger.error(f"Error in update_market_status: {e}", exc_info=True)
                last_market_update = now
            
            # BTC price (every 3s, fallback if RTDS active) - with timeout protection
            if now - last_btc_update >= config.btc_price_interval:
                try:
                    await asyncio.wait_for(core.update_btc_price(), timeout=8.0)
                except asyncio.TimeoutError:
                    logger.warning("update_btc_price timed out")
                except Exception as e:
                    logger.error(f"Error in update_btc_price: {e}", exc_info=True)
                last_btc_update = now
            
            # Account stats (every 10s) - with timeout protection
            if now - last_stats_update >= config.account_stats_interval:
                try:
                    await asyncio.wait_for(core.update_account_stats(), timeout=8.0)
                except asyncio.TimeoutError:
                    logger.warning("update_account_stats timed out")
                except Exception as e:
                    logger.error(f"Error in update_account_stats: {e}", exc_info=True)
                last_stats_update = now
            
            # Countdown (every 200ms) - with timeout protection
            if now - last_countdown_update >= config.countdown_interval:
                try:
                    await asyncio.wait_for(core.update_countdown(), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning("update_countdown timed out")
                except Exception as e:
                    logger.error(f"Error in update_countdown: {e}", exc_info=True)
                last_countdown_update = now
            
            # Analytics (every 500ms) - with timeout protection
            if now - last_analytics_update >= config.analytics_interval:
                try:
                    await asyncio.wait_for(core.update_analytics(), timeout=3.0)
                except asyncio.TimeoutError:
                    logger.warning("update_analytics timed out")
                except Exception as e:
                    logger.error(f"Error in update_analytics: {e}", exc_info=True)
                last_analytics_update = now
            
            # Small sleep to prevent busy loop
            await asyncio.sleep(0.1)
            
        except asyncio.CancelledError:
            print("[UPDATE LOOP] Cancelled, stopping...")
            logger.info("Update loop cancelled")
            break
        except Exception as e:
            logger.error(f"Error in update loop: {e}", exc_info=True)
            await asyncio.sleep(1.0)


# =============================================================================
# Application Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    global core, background_task
    
    print("[STARTUP] Starting FingerBlaster Web API...")
    logger.info("Starting FingerBlaster Web API...")
    
    # Log API key status
    if API_KEY:
        print("[STARTUP] API key authentication ENABLED")
        logger.info("API key authentication ENABLED (via FB_API_KEY env)")
    else:
        print("[STARTUP] API key authentication DISABLED (dev mode)")
        logger.info(f"API key authentication DISABLED (dev mode)")
    
    # Initialize core
    print("[STARTUP] Initializing FingerBlasterCore...")
    try:
        core = FingerBlasterCore()
        print("[STARTUP] FingerBlasterCore initialized successfully")
    except Exception as e:
        print(f"[STARTUP ERROR] Failed to initialize core: {e}")
        raise
    
    # Register callbacks for broadcasting to WebSocket clients
    print("[STARTUP] Registering callbacks...")
    core.register_callback('market_update', on_market_update)
    core.register_callback('btc_price_update', on_btc_price_update)
    core.register_callback('price_update', on_price_update)
    core.register_callback('account_stats_update', on_account_stats_update)
    core.register_callback('countdown_update', on_countdown_update)
    core.register_callback('prior_outcomes_update', on_prior_outcomes_update)
    core.register_callback('resolution', on_resolution)
    core.register_callback('log', on_log)
    core.register_callback('chart_update', on_chart_update)
    core.register_callback('analytics_update', on_analytics_update)
    
    # Start background update loop
    print("[STARTUP] Starting background update loop...")
    background_task = asyncio.create_task(run_update_loop())
    
    # Log registered routes for debugging
    routes = [f"{route.methods if hasattr(route, 'methods') else 'WS'} {route.path}" 
              for route in app.routes]
    print(f"[STARTUP] Registered routes: {', '.join(sorted(routes))}")
    logger.info(f"Registered routes: {', '.join(sorted(routes))}")
    
    print("[STARTUP] FingerBlaster Web API started successfully!")
    logger.info("FingerBlaster Web API started successfully")
    
    yield
    
    # Shutdown
    print("[SHUTDOWN] Shutting down FingerBlaster Web API...")
    logger.info("Shutting down FingerBlaster Web API...")
    
    if background_task:
        print("[SHUTDOWN] Cancelling background task...")
        background_task.cancel()
        try:
            await asyncio.wait_for(background_task, timeout=3.0)
        except asyncio.TimeoutError:
            print("[SHUTDOWN] Background task timeout, forcing cancel")
        except asyncio.CancelledError:
            pass
        print("[SHUTDOWN] Background task stopped")
    
    if core:
        print("[SHUTDOWN] Shutting down core...")
        try:
            await asyncio.wait_for(core.shutdown(), timeout=10.0)
        except asyncio.TimeoutError:
            print("[SHUTDOWN] Core shutdown timeout!")
        print("[SHUTDOWN] Core shutdown complete")
    
    logger.info("FingerBlaster Web API shutdown complete")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="FingerBlaster API",
    description="Trading API for FingerBlaster Web GUI",
    version="1.0.0",
    lifespan=lifespan,
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
origins = [origin.strip() for origin in allowed_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware for debugging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests for debugging."""
    path = request.url.path
    method = request.method
    print(f"[REQUEST] {method} {path}", flush=True)
    
    response = await call_next(request)
    
    print(f"[RESPONSE] {method} {path} -> {response.status_code}", flush=True)
    
    if response.status_code == 404:
        print(f"[404 ERROR] {method} {path} -> 404 Not Found!", flush=True)
        logger.warning(f"404 Not Found: {method} {path}")
        # Log available routes for debugging
        available_routes = [f"{r.methods if hasattr(r, 'methods') else 'WS'} {r.path}" 
                           for r in app.routes if hasattr(r, 'path')]
        print(f"[404 ERROR] Available routes: {', '.join(sorted(available_routes))}", flush=True)
        logger.warning(f"Available routes: {', '.join(sorted(available_routes))}")
    
    return response


# =============================================================================
# REST Endpoints
# =============================================================================

@app.get("/api/health", response_model=HealthResponse)
@limiter.limit("60/minute")
async def health_check(request: Request):
    """Health check endpoint."""
    market_active = False
    if core:
        market = await core.market_manager.get_market()
        market_active = market is not None
    
    return HealthResponse(
        status="ok",
        connected=core is not None,
        market_active=market_active,
    )


@app.post("/api/order", response_model=OrderResponse)
@limiter.limit("30/minute")
async def place_order(
    req: OrderRequest,
    request: Request,
    _auth: bool = Depends(verify_api_key),
):
    """Place a market order (BUY YES or BUY NO)."""
    if not core:
        raise HTTPException(status_code=503, detail="Core not initialized")
    
    # Check if market is active
    market = await core.market_manager.get_market()
    if not market:
        raise HTTPException(status_code=400, detail="No active market")
    
    # Update size if provided
    if req.size is not None:
        core.selected_size = req.size
    
    # Place the order
    await core.place_order(req.side)
    
    return OrderResponse(
        status="submitted",
        side=req.side,
        size=core.selected_size,
    )


@app.post("/api/flatten", response_model=StatusResponse)
@limiter.limit("10/minute")
async def flatten_positions(
    request: Request,
    _auth: bool = Depends(verify_api_key),
):
    """Flatten (close) all open positions."""
    if not core:
        raise HTTPException(status_code=503, detail="Core not initialized")
    
    await core.flatten()
    return StatusResponse(status="flattening")


@app.post("/api/cancel", response_model=StatusResponse)
@limiter.limit("20/minute")
async def cancel_all_orders(
    request: Request,
    _auth: bool = Depends(verify_api_key),
):
    """Cancel all pending orders."""
    if not core:
        raise HTTPException(status_code=503, detail="Core not initialized")
    
    await core.cancel_all()
    return StatusResponse(status="cancelling")


@app.post("/api/size", response_model=dict)
@limiter.limit("60/minute")
async def adjust_size(
    req: SizeRequest,
    request: Request,
    _auth: bool = Depends(verify_api_key),
):
    """Adjust the order size up or down."""
    if not core:
        raise HTTPException(status_code=503, detail="Core not initialized")
    
    if req.action == "up":
        core.size_up()
    else:
        core.size_down()
    
    return {"status": "ok", "size": core.selected_size}


@app.post("/api/discover-market", response_model=StatusResponse)
@limiter.limit("10/minute")
async def discover_market(
    request: Request,
    _auth: bool = Depends(verify_api_key),
):
    """Manually trigger market discovery."""
    print("[API] /api/discover-market endpoint called!")
    logger.info("Discover market endpoint called")
    
    if not core:
        print("[API ERROR] Core not initialized!")
        logger.error("Core not initialized when discover_market was called")
        raise HTTPException(status_code=503, detail="Core not initialized")
    
    try:
        print("[API] Starting market discovery...")
        logger.info("Starting market discovery...")
        await core.update_market_status()
        
        # Check if market was found
        market = await core.market_manager.get_market()
        if market:
            print(f"[API] Market discovery successful: market_id={market.get('market_id', 'unknown')}")
            logger.info(f"Market discovery successful: market_id={market.get('market_id', 'unknown')}")
        else:
            print("[API] Market discovery completed but no market was found")
            logger.warning("Market discovery completed but no market was found")
        
        return StatusResponse(status="market discovery triggered")
    except Exception as e:
        print(f"[API ERROR] Market discovery error: {e}")
        logger.error(f"Market discovery error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Market discovery failed: {str(e)}")


async def _gather_full_state() -> dict:
    """Helper function to gather the full current state.
    
    This can be called from both the REST endpoint and WebSocket handler.
    """
    if not core:
        raise HTTPException(status_code=503, detail="Core not initialized")
    
    # Gather current state
    market = await core.market_manager.get_market()
    token_map = await core.market_manager.get_token_map()
    prices = await core.market_manager.calculate_mid_price()
    btc_history = await core.history_manager.get_btc_history()
    yes_history = await core.history_manager.get_yes_history()
    
    # Extract price to beat - check multiple possible keys
    strike_value = None
    if market:
        # Try different possible keys for price to beat
        strike_value = market.get('price_to_beat') or market.get('strike') or None
        if strike_value:
            strike_value = str(strike_value).strip()
            if strike_value in ('N/A', 'None', '', 'Dynamic', 'Pending'):
                strike_value = None
        if not strike_value:
            logger.debug(f"Market exists but no price_to_beat found. Market keys: {list(market.keys())}")
    
    # Get account balances (async methods with timeout)
    balance = 0.0
    yes_bal = 0.0
    no_bal = 0.0
    try:
        balance = await asyncio.wait_for(
            core.connector.get_usdc_balance(),
            timeout=3.0
        )
        if token_map:
            y_id = token_map.get('Up')
            n_id = token_map.get('Down')
            if y_id:
                yes_bal = await asyncio.wait_for(
                    core.connector.get_token_balance(y_id),
                    timeout=3.0
                )
            if n_id:
                no_bal = await asyncio.wait_for(
                    core.connector.get_token_balance(n_id),
                    timeout=3.0
                )
    except asyncio.TimeoutError:
        logger.warning("Balance fetch timeout, using cached values")
    except Exception as e:
        logger.debug(f"Error fetching balances: {e}")
    
    # Build response
    yes_price, no_price, best_bid, best_ask = prices
    yes_spread, no_spread = FingerBlasterCore.calculate_spreads(best_bid, best_ask)
    
    state = {
        "market": {
            "active": market is not None,
            "priceToBeat": strike_value if strike_value else (None if market else None),
            "endDate": market.get('end_date') if market else None,
        },
        "prices": {
            "yesPrice": yes_price,
            "noPrice": no_price,
            "bestBid": best_bid,
            "bestAsk": best_ask,
            "yesSpread": yes_spread,
            "noSpread": no_spread,
        },
        "account": {
            "balance": balance,
            "yesBalance": yes_bal,
            "noBalance": no_bal,
            "selectedSize": core.selected_size,
            "avgEntryYes": core.avg_entry_price_up,
            "avgEntryNo": core.avg_entry_price_down,
        },
        "btcPrice": btc_history[-1] if btc_history else 0.0,
        "priorOutcomes": core.displayed_prior_outcomes,
    }
    
    return state


@app.get("/api/state")
@limiter.limit("30/minute")
async def get_full_state(
    request: Request,
    _auth: bool = Depends(verify_api_key),
):
    """Get the full current state (for initial sync on connect)."""
    return await _gather_full_state()


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data streaming."""
    await manager.connect(websocket)
    
    try:
        # Send initial state on connect - don't let failures close the connection
        if core:
            try:
                # Add timeout to prevent blocking on WebSocket connect
                state = await asyncio.wait_for(_gather_full_state(), timeout=5.0)
                await websocket.send_text(json.dumps({
                    "event": "initial_state",
                    "data": state,
                }))
                logger.debug("Initial state sent successfully")
            except asyncio.TimeoutError:
                logger.warning("Initial state gathering timed out, sending minimal state")
                # Send minimal state if gathering times out
                try:
                    await websocket.send_text(json.dumps({
                        "event": "initial_state",
                        "data": {
                            "market": {"active": False},
                            "prices": {"yesPrice": 0.5, "noPrice": 0.5},
                            "account": {"balance": 0.0, "yesBalance": 0.0, "noBalance": 0.0, "selectedSize": 0.0},
                            "btcPrice": 0.0,
                            "priorOutcomes": [],
                        },
                    }))
                except Exception as send_error:
                    logger.error(f"Failed to send minimal state: {send_error}")
            except Exception as e:
                logger.error(f"Error gathering initial state: {e}", exc_info=True)
                # Try to send minimal state even on error
                try:
                    await websocket.send_text(json.dumps({
                        "event": "initial_state",
                        "data": {
                            "market": {"active": False},
                            "prices": {"yesPrice": 0.5, "noPrice": 0.5},
                            "account": {"balance": 0.0, "yesBalance": 0.0, "noBalance": 0.0, "selectedSize": 0.0},
                            "btcPrice": 0.0,
                            "priorOutcomes": [],
                        },
                    }))
                except Exception as send_error:
                    logger.error(f"Failed to send fallback state: {send_error}")
        
        # Keep connection alive and handle client messages
        while True:
            try:
                data = await websocket.receive_text()
                
                # Handle ping/pong
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "ping":
                        await websocket.send_text(json.dumps({"event": "pong"}))
                    elif msg.get("type") == "sync_request":
                        # Client requests full state sync
                        if core:
                            try:
                                state = await asyncio.wait_for(_gather_full_state(), timeout=5.0)
                                await websocket.send_text(json.dumps({
                                    "event": "initial_state",
                                    "data": state,
                                }))
                            except asyncio.TimeoutError:
                                logger.warning("State sync request timed out")
                            except Exception as e:
                                logger.error(f"Error in sync_request: {e}", exc_info=True)
                except json.JSONDecodeError:
                    logger.debug(f"Invalid JSON received: {data[:100]}")
                    continue
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected normally")
                break
            except Exception as e:
                # Log but don't break the connection on message errors
                logger.error(f"WebSocket message error: {e}", exc_info=True)
                # Continue the loop to keep connection alive
                await asyncio.sleep(0.1)
    except Exception as e:
        logger.error(f"WebSocket endpoint error: {e}", exc_info=True)
    finally:
        await manager.disconnect(websocket)


# =============================================================================
# Entry Point
# =============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()

