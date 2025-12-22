"""Core business logic controller - UI agnostic."""

import asyncio
import json
import logging
import os
import time
from collections.abc import Callable, Awaitable
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple

import pandas as pd

from connectors.polymarket import PolymarketConnector
from src.config import AppConfig
from src.engine import MarketDataManager, HistoryManager, WebSocketManager, OrderExecutor

logger = logging.getLogger("FingerBlaster")


class FingerBlasterCore:
    """Shared business logic controller that both UIs can use."""
    
    def __init__(self):
        """Initialize the core controller."""
        self.config = AppConfig()
        self.connector = PolymarketConnector()
        
        # Initialize managers
        self.market_manager = MarketDataManager(self.config)
        self.history_manager = HistoryManager(self.config)
        self.order_executor = OrderExecutor(self.config, self.connector)
        self.ws_manager = WebSocketManager(
            self.config,
            self.market_manager,
            self._on_ws_message
        )
        
        # State
        self.resolution_shown = False
        self.last_resolution: Optional[str] = None
        self.prior_outcomes: List[Dict[str, Any]] = []
        self.displayed_prior_outcomes: List[str] = []
        self.last_chart_update: float = 0.0
        self.selected_size: float = 1.0
        
        # Callbacks for UI updates
        self._callbacks: Dict[str, List[Callable]] = {
            'market_update': [],
            'btc_price_update': [],
            'price_update': [],
            'account_stats_update': [],
            'countdown_update': [],
            'prior_outcomes_update': [],
            'resolution': [],
            'log': [],
            'chart_update': [],
        }
        
        # Load prior outcomes
        self._load_prior_outcomes()
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for a specific event.
        
        Args:
            event: Event name ('market_update', 'btc_price_update', etc.)
            callback: Callback function to call when event occurs
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _emit(self, event: str, *args, **kwargs) -> None:
        """Emit an event to all registered callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    # Try to get the running event loop, create task if possible
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(callback(*args, **kwargs))
                    except RuntimeError:
                        # No running loop - this can happen in Qt
                        # Try to get any event loop
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                loop.create_task(callback(*args, **kwargs))
                            else:
                                # Loop exists but not running - schedule it
                                asyncio.ensure_future(callback(*args, **kwargs), loop=loop)
                        except (RuntimeError, AttributeError):
                            # No loop available - this will be handled by Qt integration
                            # Just log a warning, the Qt layer will handle it
                            logger.debug(f"No event loop available for async callback {event}")
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback for {event}: {e}")
    
    def log_msg(self, message: str) -> None:
        """Log message and emit to UI."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        logger.info(message)
        self._emit('log', f"[{timestamp}] {message}")
    
    async def _on_ws_message(self, item: Dict[str, Any]) -> None:
        """Handle WebSocket message by recalculating price."""
        await self._recalc_price()
    
    async def _recalc_price(self) -> None:
        """Recalculate mid price and update UI."""
        yes_price, no_price, best_bid, best_ask = await self.market_manager.calculate_mid_price()
        
        # Emit price update
        self._emit('price_update', yes_price, no_price, best_bid, best_ask)
        
        # Update history
        market_start_time = await self.market_manager.get_market_start_time()
        if market_start_time:
            now = pd.Timestamp.now(tz='UTC')
            elapsed = (now - market_start_time).total_seconds()
            await self.history_manager.add_price_point(elapsed, yes_price, market_start_time)
        
        # Emit chart update (throttled)
        now = time.time()
        if now - self.last_chart_update >= self.config.chart_update_throttle_seconds:
            self.last_chart_update = now
            history = await self.history_manager.get_yes_history()
            self._emit('chart_update', history)
    
    async def update_market_status(self) -> None:
        """Update market status and search for new markets."""
        market = await self.market_manager.get_market()
        if not market:
            # Search for new market
            try:
                new_market = await asyncio.to_thread(self.connector.get_active_market)
                if new_market:
                    success = await self.market_manager.set_market(new_market)
                    if success:
                        await self.history_manager.clear_yes_history()
                        self.log_msg(f"Market Found: {new_market.get('strike_price', 'N/A')}")
                        await self.ws_manager.start()
                        
                        # Emit market update
                        strike = str(new_market.get('strike_price', 'N/A'))
                        ends = self._format_ends(new_market.get('end_date', 'N/A'))
                        self._emit('market_update', strike, ends)
                        
                        # Reset countdown display when new market loads
                        # This will be updated by the countdown timer, but we can trigger an immediate update
                        await self.update_countdown()
                        
                        # Re-check prior outcomes for the new market
                        await self._check_and_add_prior_outcomes()
            except Exception as e:
                logger.error(f"Error searching for market: {e}")
        
        await self.check_if_market_expired()
    
    async def update_btc_price(self) -> None:
        """Update BTC price and refresh chart."""
        try:
            price = await asyncio.to_thread(self.connector.get_btc_price)
            if price and isinstance(price, (int, float)) and price > 0:
                btc_price = float(price)
                await self.history_manager.add_btc_price(btc_price)
                
                # Emit BTC price update
                self._emit('btc_price_update', btc_price)
                
                # Emit BTC chart update
                prices = await self.history_manager.get_btc_history()
                if len(prices) >= 2:
                    market = await self.market_manager.get_market()
                    strike_val = None
                    if market:
                        strike_str = str(market.get('strike_price', ''))
                        strike_val = self._parse_strike(strike_str)
                    self._emit('chart_update', prices, strike_val, 'btc')
        except Exception as e:
            logger.error(f"Error updating BTC price: {e}")
    
    async def update_account_stats(self) -> None:
        """Update account statistics."""
        try:
            token_map = await self.market_manager.get_token_map()
            
            def get_stats():
                bal = self.connector.get_usdc_balance()
                yes_bal = 0.0
                no_bal = 0.0
                if token_map:
                    y_id = token_map.get('YES')
                    n_id = token_map.get('NO')
                    if y_id:
                        yes_bal = self.connector.get_token_balance(y_id)
                    if n_id:
                        no_bal = self.connector.get_token_balance(n_id)
                return float(bal or 0.0), float(yes_bal or 0.0), float(no_bal or 0.0)
            
            bal, y, n = await asyncio.to_thread(get_stats)
            self._emit('account_stats_update', bal, y, n, self.selected_size)
        except Exception as e:
            logger.error(f"Error updating account stats: {e}")
    
    async def update_countdown(self) -> None:
        """Update the countdown timer."""
        market = await self.market_manager.get_market()
        if not market:
            # If no market, show N/A but don't emit to avoid glitching
            return
        
        try:
            end_str = market.get('end_date')
            if not end_str:
                return
            
            dt_end = pd.Timestamp(end_str)
            if dt_end.tz is None:
                dt_end = dt_end.tz_localize('UTC')
            
            now = pd.Timestamp.now(tz='UTC')
            diff = dt_end - now
            total_seconds = diff.total_seconds()
            
            if total_seconds < 0:
                # Market has expired - show EXPIRED and keep showing it
                time_str = "EXPIRED"
            else:
                # Calculate time remaining with seconds precision
                secs = int(total_seconds)
                mins = secs // 60
                remaining_secs = secs % 60
                time_str = f"{mins:02d}:{remaining_secs:02d}"
            
            self._emit('countdown_update', time_str)
        except Exception as e:
            logger.debug(f"Error updating countdown: {e}")
    
    async def check_if_market_expired(self) -> None:
        """Check if market has expired and show resolution."""
        market = await self.market_manager.get_market()
        if not market:
            return
        
        try:
            end_str = market.get('end_date')
            if not end_str:
                return
            
            end_dt = pd.Timestamp(end_str)
            if end_dt.tz is None:
                end_dt = end_dt.tz_localize('UTC')
            
            now = pd.Timestamp.now(tz='UTC')
            if now > end_dt and not self.resolution_shown:
                self.resolution_shown = True
                await self._show_resolution()
                await asyncio.sleep(self.config.resolution_overlay_duration)
                await self._reset_market_after_resolution()
        except Exception as e:
            logger.error(f"Error checking market expiry: {e}")
    
    async def _show_resolution(self) -> None:
        """Calculate and emit resolution."""
        try:
            market = await self.market_manager.get_market()
            if not market:
                return
            
            # Get BTC price from latest history
            btc_history = await self.history_manager.get_btc_history()
            btc_price = btc_history[-1] if btc_history else 0.0
            
            strike_str = str(market.get('strike_price', '')).replace(',', '').replace('$', '').strip()
            
            if strike_str and strike_str != "N/A":
                try:
                    strike_val = float(strike_str)
                    resolution = "YES" if btc_price >= strike_val else "NO"
                except (ValueError, TypeError):
                    resolution = "YES"
            else:
                resolution = "YES"
            
            self.last_resolution = resolution
            self.log_msg(f"Market Resolved: {resolution} (BTC: ${btc_price:,.2f} vs Strike: {strike_str})")
            self._emit('resolution', resolution)
        except Exception as e:
            logger.error(f"Error showing resolution: {e}")
    
    async def _reset_market_after_resolution(self) -> None:
        """Reset market state after resolution."""
        try:
            if self.last_resolution:
                # Get market before clearing it
                market = await self.market_manager.get_market()
                await self._add_prior_outcome(self.last_resolution, market)
                self.last_resolution = None
            
            self.log_msg("Clearing expired market...")
            await self.market_manager.clear_market()
            await self.history_manager.clear_yes_history()
            await self.ws_manager.stop()
            self.resolution_shown = False
            self._emit('resolution', None)  # Hide overlay
        except Exception as e:
            logger.error(f"Error resetting market: {e}")
    
    async def _check_and_add_prior_outcomes(self) -> None:
        """Check if prior outcomes should be displayed based on consecutive market timing."""
        try:
            market = await self.market_manager.get_market()
            if not market or not self.prior_outcomes:
                self.displayed_prior_outcomes = []
                self._update_prior_outcomes_display()
                return
            
            # Get current market start time
            market_start_time = await self.market_manager.get_market_start_time()
            if not market_start_time:
                self.displayed_prior_outcomes = []
                self._update_prior_outcomes_display()
                return
            
            # Calculate what the immediately preceding market's end time should be
            expected_previous_market_end = market_start_time - pd.Timedelta(
                minutes=self.config.market_duration_minutes
            )
            
            # Tolerance for time matching (1 minute to account for slight variations)
            tolerance_seconds = 60
            
            # Filter outcomes to only include consecutive ones
            consecutive_outcomes = []
            expected_end_time = expected_previous_market_end
            
            # Iterate backwards through prior_outcomes
            for outcome_entry in reversed(self.prior_outcomes):
                if isinstance(outcome_entry, str):
                    break
                
                timestamp_str = outcome_entry.get('timestamp')
                if not timestamp_str:
                    break
                
                try:
                    outcome_timestamp = pd.Timestamp(timestamp_str)
                    if outcome_timestamp.tz is None:
                        outcome_timestamp = outcome_timestamp.tz_localize('UTC')
                    
                    time_diff = abs((outcome_timestamp - expected_end_time).total_seconds())
                    
                    if time_diff <= tolerance_seconds:
                        consecutive_outcomes.insert(0, outcome_entry.get('outcome', ''))
                        expected_end_time = outcome_timestamp - pd.Timedelta(
                            minutes=self.config.market_duration_minutes
                        )
                    else:
                        break
                        
                except Exception as e:
                    logger.debug(f"Error processing outcome timestamp: {e}")
                    break
            
            self.displayed_prior_outcomes = consecutive_outcomes
            self._update_prior_outcomes_display()
            
            if consecutive_outcomes:
                self.log_msg(f"Displaying {len(consecutive_outcomes)} consecutive prior outcome(s)")
                    
        except Exception as e:
            logger.debug(f"Error in _check_and_add_prior_outcomes: {e}")
            self.displayed_prior_outcomes = []
            self._update_prior_outcomes_display()
    
    def _update_prior_outcomes_display(self) -> None:
        """Update prior outcomes display."""
        outcomes_to_display = self.displayed_prior_outcomes if self.displayed_prior_outcomes else [
            (outcome_entry if isinstance(outcome_entry, str) else outcome_entry.get('outcome', ''))
            for outcome_entry in self.prior_outcomes
        ]
        self._emit('prior_outcomes_update', outcomes_to_display)
    
    async def _add_prior_outcome(self, outcome: str, market: Optional[Dict[str, Any]] = None) -> None:
        """Add outcome to prior outcomes list with timestamp."""
        outcome_upper = outcome.upper()
        if outcome_upper in ("YES", "NO"):
            if market is None:
                market = await self.market_manager.get_market()
            
            timestamp = None
            if market and market.get('end_date'):
                try:
                    end_dt = pd.Timestamp(market.get('end_date'))
                    if end_dt.tz is None:
                        end_dt = end_dt.tz_localize('UTC')
                    timestamp = end_dt.isoformat()
                except Exception as e:
                    logger.debug(f"Error getting timestamp for outcome: {e}")
            
            if not timestamp:
                timestamp = pd.Timestamp.now(tz='UTC').isoformat()
            
            outcome_entry = {
                'outcome': outcome_upper,
                'timestamp': timestamp
            }
            self.prior_outcomes.append(outcome_entry)
            if len(self.prior_outcomes) > self.config.max_prior_outcomes:
                self.prior_outcomes.pop(0)
            self._save_prior_outcomes()
            self._update_prior_outcomes_display()
    
    def _load_prior_outcomes(self) -> None:
        """Load prior outcomes from file."""
        try:
            if os.path.exists(self.config.prior_outcomes_file):
                with open(self.config.prior_outcomes_file, 'r') as f:
                    data = json.load(f)
                    outcomes = data.get('outcomes', [])
                    
                    normalized_outcomes = []
                    for outcome in outcomes:
                        if isinstance(outcome, str):
                            normalized_outcomes.append({
                                'outcome': outcome,
                                'timestamp': None
                            })
                        elif isinstance(outcome, dict):
                            normalized_outcomes.append(outcome)
                    
                    self.prior_outcomes = normalized_outcomes[:self.config.max_prior_outcomes]
        except Exception as e:
            logger.debug(f"Error loading prior outcomes: {e}")
            self.prior_outcomes = []
    
    def _save_prior_outcomes(self) -> None:
        """Save prior outcomes to file."""
        try:
            os.makedirs(self.config.data_dir, exist_ok=True)
            with open(self.config.prior_outcomes_file, 'w') as f:
                json.dump({'outcomes': self.prior_outcomes}, f)
        except Exception as e:
            logger.debug(f"Error saving prior outcomes: {e}")
    
    def _format_ends(self, end_str: str) -> str:
        """Format end date to PST."""
        try:
            dt = pd.Timestamp(end_str)
            if dt.tz is None:
                dt = dt.tz_localize('UTC')
            dt_local = dt.tz_convert('US/Pacific')
            return dt_local.strftime("%H:%M:%S PST")
        except Exception as e:
            logger.debug(f"Error formatting end date: {e}")
            return str(end_str)
    
    def _parse_strike(self, strike_str: str) -> Optional[float]:
        """Parse strike price string to float."""
        try:
            clean_strike = strike_str.replace(',', '').replace('$', '').strip()
            if clean_strike and clean_strike != "N/A":
                return float(clean_strike)
        except (ValueError, AttributeError):
            pass
        return None
    
    def size_up(self) -> None:
        """Increase order size."""
        self.selected_size += self.config.size_increment
        self.log_msg(f"Size increased to ${self.selected_size:.2f}")
    
    def size_down(self) -> None:
        """Decrease order size."""
        if self.selected_size > self.config.min_order_size:
            self.selected_size -= self.config.size_increment
            self.log_msg(f"Size decreased to ${self.selected_size:.2f}")
    
    async def place_order(self, side: str) -> None:
        """Place an order."""
        try:
            size = self.selected_size
            self.log_msg(f"Order: BUY {side} (${size:.2f})")
            
            token_map = await self.market_manager.get_token_map()
            resp = await self.order_executor.execute_order(side, size, token_map)
            
            if resp and resp.get('orderID'):
                self.log_msg(f"Order FILLED: {resp['orderID'][:10]}...")
                await self.update_account_stats()
            else:
                self.log_msg("Order FAILED")
        except Exception as e:
            self.log_msg(f"Execution Error: {e}")
            logger.error(f"Order placement error: {e}")
    
    async def flatten(self) -> None:
        """Flatten all positions."""
        self.log_msg("Action: FLATTEN")
        token_map = await self.market_manager.get_token_map()
        if not token_map:
            self.log_msg("Error: Token map not ready.")
            return
        
        results = await self.order_executor.flatten_positions(token_map)
        if results:
            self.log_msg(f"Flatten completed. {len(results)} orders processed.")
        else:
            self.log_msg("Flatten completed. No orders to process.")
        
        await self.update_account_stats()
    
    async def cancel_all(self) -> None:
        """Cancel all pending orders."""
        self.log_msg("Action: CANCEL ALL")
        success = await self.order_executor.cancel_all_orders()
        if success:
            self.log_msg("All orders cancelled.")
        else:
            self.log_msg("Cancel operation failed.")
    
    async def shutdown(self) -> None:
        """Handle graceful shutdown."""
        self.log_msg("Initiating graceful shutdown...")
        await self.ws_manager.stop()

