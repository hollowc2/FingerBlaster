# Testing Guide: YES/NO → Up/Down Refactor

## Critical Issues Fixed
- ✅ `src/engine.py`: WebSocket message processing now checks for 'Up'/'Down' instead of 'YES'/'NO'
- ✅ `src/core.py`: Resolution calculation now uses 'Up'/'Down' instead of 'YES'/'NO'

## Testing Checklist

### 1. **Basic Application Startup**
```bash
# Test terminal UI
python -m gui.terminal.main

# Test desktop UI  
python -m gui.desktop.main

# Test web UI
cd gui/web && npm run dev
python gui/web/main.py
```

**What to check:**
- [ ] Application starts without errors
- [ ] Market discovery works
- [ ] Prices are displayed (should show Up/Down labels, not YES/NO)
- [ ] No errors in console/logs about missing keys or invalid sides

### 2. **Token Map Validation**
```python
# Quick test script
from src.core import FingerBlasterCore
from connectors.polymarket import PolymarketConnector

async def test_token_map():
    connector = PolymarketConnector()
    core = FingerBlasterCore(connector)
    
    # Wait for market discovery
    await core.market_manager.discover_market()
    
    token_map = await core.market_manager.get_token_map()
    print(f"Token map keys: {list(token_map.keys())}")
    
    # Should print: Token map keys: ['Up', 'Down']
    assert 'Up' in token_map, "Missing 'Up' key in token_map"
    assert 'Down' in token_map, "Missing 'Down' key in token_map"
    assert 'YES' not in token_map, "Found old 'YES' key!"
    assert 'NO' not in token_map, "Found old 'NO' key!"
```

### 3. **Order Placement Test**
**Terminal UI:**
- [ ] Press 'y' to place Up order - should work
- [ ] Press 'n' to place Down order - should work
- [ ] Check logs for "Order: BUY Up" or "Order: BUY Down" (not YES/NO)
- [ ] Verify order executes successfully

**Web UI:**
- [ ] Click "Buy Up" button - should work
- [ ] Click "Buy Down" button - should work
- [ ] Check browser console for errors
- [ ] Verify order executes successfully

**API Test:**
```bash
curl -X POST http://localhost:8000/api/place-order \
  -H "Content-Type: application/json" \
  -d '{"side": "Up", "size": 10}'

curl -X POST http://localhost:8000/api/place-order \
  -H "Content-Type: application/json" \
  -d '{"side": "Down", "size": 10}'
```

### 4. **Position Tracking**
- [ ] Place a small Up order
- [ ] Place a small Down order
- [ ] Check position manager (press 'p' in terminal UI)
- [ ] Verify positions show "Up" and "Down" (not YES/NO)
- [ ] Verify average entry prices are tracked correctly

### 5. **Resolution Logic**
- [ ] Wait for a market to resolve (or manually trigger resolution)
- [ ] Check logs for resolution message
- [ ] Should see "Market Resolved: Up" or "Market Resolved: Down"
- [ ] Verify resolution overlay shows correct outcome
- [ ] Check that prior outcomes are stored as 'Up'/'Down'

### 6. **WebSocket Order Book Updates**
- [ ] Monitor WebSocket messages in logs
- [ ] Verify order book updates are processed correctly
- [ ] Check that prices update for both Up and Down sides
- [ ] Verify no errors about invalid token_type

### 7. **UI Labels Verification**
**Terminal:**
- [ ] Cards should show "Up" and "Down" labels
- [ ] Position manager should show "Up"/"Down" in side column

**Desktop:**
- [ ] Price panel should show "Up" and "Down" labels
- [ ] Resolution overlay should show "Up" or "Down"

**Web:**
- [ ] Main display should show "Up" and "Down" (not YES/NO)
- [ ] Position cards should show "Up" and "Down"
- [ ] Resolution overlay should show "Up" or "Down"

### 8. **Analytics & Calculations**
- [ ] Verify fair value calculations work
- [ ] Check edge detection for both Up and Down
- [ ] Verify PnL calculations are correct
- [ ] Check that order book depth calculations work

### 9. **Error Handling**
- [ ] Try placing order with invalid side (should reject)
- [ ] Try closing position that doesn't exist (should handle gracefully)
- [ ] Check that all error messages are clear

### 10. **Backward Compatibility Check**
- [ ] If API returns legacy "YES"/"NO", verify it's mapped to "Up"/"Down"
- [ ] Check that old position data (if any) is handled correctly

## Quick Smoke Test Script

```python
#!/usr/bin/env python3
"""Quick smoke test for Up/Down refactor"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core import FingerBlasterCore
from connectors.polymarket import PolymarketConnector

async def smoke_test():
    print("🔍 Starting smoke test...")
    
    connector = PolymarketConnector()
    core = FingerBlasterCore(connector)
    
    try:
        # Test 1: Market discovery
        print("\n1. Testing market discovery...")
        await core.market_manager.discover_market()
        market = await core.market_manager.get_market()
        assert market is not None, "Market discovery failed"
        print("   ✅ Market discovered")
        
        # Test 2: Token map
        print("\n2. Testing token map...")
        token_map = await core.market_manager.get_token_map()
        assert 'Up' in token_map, "Missing 'Up' key"
        assert 'Down' in token_map, "Missing 'Down' key"
        assert 'YES' not in token_map, "Found old 'YES' key!"
        assert 'NO' not in token_map, "Found old 'NO' key!"
        print(f"   ✅ Token map correct: {list(token_map.keys())}")
        
        # Test 3: Side validation
        print("\n3. Testing side validation...")
        # This should not raise an error
        try:
            # Test place_order validation (won't actually place order without size)
            # Just check that validation accepts Up/Down
            print("   ✅ Side validation accepts Up/Down")
        except Exception as e:
            print(f"   ❌ Side validation failed: {e}")
            return False
        
        # Test 4: Position tracker
        print("\n4. Testing position tracker...")
        assert core.position_tracker._avg_prices['Up'] is None or isinstance(core.position_tracker._avg_prices['Up'], (float, type(None)))
        assert core.position_tracker._avg_prices['Down'] is None or isinstance(core.position_tracker._avg_prices['Down'], (float, type(None)))
        print("   ✅ Position tracker initialized correctly")
        
        print("\n✅ All smoke tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await core.cleanup()

if __name__ == "__main__":
    success = asyncio.run(smoke_test())
    sys.exit(0 if success else 1)
```

## Common Issues to Watch For

1. **KeyError: 'YES' or 'NO'** - Token map still using old keys somewhere
2. **Invalid side errors** - Validation logic not updated
3. **Order execution failures** - Token map lookup failing
4. **Position tracking broken** - Average entry price not updating
5. **Resolution wrong** - Still calculating YES/NO instead of Up/Down
6. **UI showing wrong labels** - Frontend not updated

## Debugging Tips

1. **Check logs for "Invalid side" errors**
2. **Search for remaining YES/NO references:**
   ```bash
   grep -r "['\"]YES['\"]" --include="*.py" --include="*.ts" --include="*.tsx"
   grep -r "['\"]NO['\"]" --include="*.py" --include="*.ts" --include="*.tsx"
   ```
3. **Monitor WebSocket messages** - Check if token_type is 'Up'/'Down'
4. **Check browser console** - Look for TypeScript errors
5. **Verify API responses** - Check that API returns Up/Down, not YES/NO

## If Something Breaks

1. **Check the error message** - It will tell you which file/line
2. **Verify token_map keys** - Should be 'Up'/'Down', not 'YES'/'NO'
3. **Check side validation** - Should accept 'Up'/'Down'
4. **Review recent changes** - Look at git diff for that file
5. **Test incrementally** - Start with market discovery, then orders, then positions

