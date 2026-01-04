#!/usr/bin/env python3
"""Debug script to check WebSocket and token_map issues"""

import asyncio
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core import FingerBlasterCore
from connectors.polymarket import PolymarketConnector

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FingerBlaster")

async def debug_check():
    print("=" * 60)
    print("DEBUGGING WEBSOCKET AND TOKEN_MAP")
    print("=" * 60)
    
    core = FingerBlasterCore()
    
    try:
        # Step 1: Check market discovery
        print("\n1. Checking market discovery...")
        await core.update_market_status()
        market = await core.market_manager.get_market()
        
        if not market:
            print("   ❌ No market found!")
            return
        
        print(f"   ✅ Market found: {market.get('market_id', 'unknown')}")
        
        # Step 2: Check token_map
        print("\n2. Checking token_map...")
        token_map = await core.market_manager.get_token_map()
        
        if not token_map:
            print("   ❌ token_map is empty!")
            return
        
        print(f"   ✅ token_map keys: {list(token_map.keys())}")
        for key, value in token_map.items():
            print(f"      - {key}: {value}")
        
        # Step 3: Check raw_books initialization
        print("\n3. Checking raw_books initialization...")
        raw_books = core.market_manager.raw_books
        print(f"   raw_books keys: {list(raw_books.keys())}")
        for key in raw_books.keys():
            print(f"      - {key}: {len(raw_books[key]['bids'])} bids, {len(raw_books[key]['asks'])} asks")
        
        # Step 4: Check WebSocket status
        print("\n4. Checking WebSocket status...")
        ws_manager = core.ws_manager
        if ws_manager.connection_task and not ws_manager.connection_task.done():
            print("   ✅ WebSocket connection task is running")
        else:
            print("   ⚠️  WebSocket connection task not running")
            print("   Starting WebSocket...")
            await ws_manager.start()
            await asyncio.sleep(2)  # Give it time to connect
        
        # Step 5: Check if WebSocket is connected
        if ws_manager._ws:
            print("   ✅ WebSocket is connected")
        else:
            print("   ❌ WebSocket is not connected")
        
        # Step 6: Try to calculate prices
        print("\n5. Testing price calculation...")
        prices = await core.market_manager.calculate_mid_price()
        yes_price, no_price, best_bid, best_ask = prices
        print(f"   yes_price: {yes_price:.4f}")
        print(f"   no_price: {no_price:.4f}")
        print(f"   best_bid: {best_bid:.4f}")
        print(f"   best_ask: {best_ask:.4f}")
        
        if yes_price == 0.5 and no_price == 0.5 and best_bid == 0.0 and best_ask == 1.0:
            print("   ⚠️  Prices are default values - order book is likely empty")
        else:
            print("   ✅ Prices calculated successfully")
        
        # Step 7: Wait a bit and check for WebSocket messages
        print("\n6. Waiting 5 seconds for WebSocket messages...")
        print("   (Check logs above for 'Processing WebSocket message' or 'not found in token_map')")
        await asyncio.sleep(5)
        
        # Check prices again
        prices2 = await core.market_manager.calculate_mid_price()
        yes_price2, no_price2, best_bid2, best_ask2 = prices2
        if prices2 != prices:
            print(f"   ✅ Prices changed! New prices: yes={yes_price2:.4f}, no={no_price2:.4f}")
        else:
            print(f"   ⚠️  Prices unchanged - WebSocket may not be receiving updates")
        
        print("\n" + "=" * 60)
        print("DEBUG COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during debug: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup if method exists
        if hasattr(core, 'cleanup'):
            await core.cleanup()

if __name__ == "__main__":
    asyncio.run(debug_check())

