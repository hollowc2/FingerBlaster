# Verification Checklist: Strike Price → Price to Beat Refactor

## Quick Verification Steps

### 1. **Check for Syntax Errors**
```bash
# Python syntax check
python -m py_compile src/core.py connectors/polymarket.py src/analytics.py

# TypeScript check (if you have tsc)
cd gui/web && npm run type-check  # if available
```

### 2. **Run the Application**
Test each GUI to ensure they start without errors:

**Terminal GUI:**
```bash
python gui/terminal/main.py
```
- ✅ App should start without errors
- ✅ "PRICE TO BEAT" should display instead of "STRIKE PRICE"
- ✅ Price to beat value should show correctly
- ✅ Delta calculation should work (BTC price - Price to Beat)

**Desktop GUI:**
```bash
python gui/desktop/main.py
```
- ✅ App should start without errors
- ✅ "PRICE TO BEAT" label should display
- ✅ Chart should show price to beat line

**Web GUI:**
```bash
cd gui/web && python main.py
# Then open browser to http://localhost:8000
```
- ✅ Web interface should load
- ✅ "Price to Beat" should display in UI
- ✅ WebSocket should connect and receive data

### 3. **Check Logs for Errors**
Watch the console/logs for:
- ❌ `AttributeError: 'FingerBlasterCore' object has no attribute '_parse_strike'`
- ❌ `KeyError: 'strike_price'`
- ❌ Any errors about missing functions or keys

### 4. **Functional Tests**

**Market Data Flow:**
- [ ] Market discovery works
- [ ] Price to beat is extracted from Polymarket API
- [ ] Price to beat displays correctly in all GUIs
- [ ] Dynamic price to beat resolution works (if applicable)

**Analytics:**
- [ ] Delta calculation works (BTC - Price to Beat)
- [ ] Fair value calculations work
- [ ] Edge calculations work
- [ ] Z-score/sigma calculations work

**Charts:**
- [ ] BTC chart shows price to beat line
- [ ] Chart updates correctly

**Resolution:**
- [ ] Market resolution logic works (BTC >= Price to Beat = Up)

### 5. **Search for Remaining Issues**
```bash
# Search for any remaining 'strike_price' dictionary keys (should be none)
grep -r "strike_price" --include="*.py" --include="*.ts" --include="*.tsx"

# Search for old function names (should be none)
grep -r "_parse_strike\|_extract_strike_price\|_resolve_dynamic_strike" --include="*.py"
```

### 6. **WebSocket/API Verification**
- [ ] WebSocket events send `priceToBeat` (not `strike`)
- [ ] API responses include `priceToBeat` field
- [ ] Frontend receives and displays `priceToBeat` correctly

## Known Issues Fixed
- ✅ Fixed `_parse_strike()` call in `_show_resolution()` → now uses `_parse_price_to_beat()`

## What to Watch For

**Runtime Errors:**
- AttributeError (function not found)
- KeyError (dictionary key not found)
- TypeError (wrong parameter names)

**Data Flow Issues:**
- Price to beat not displaying
- Delta showing as 0 or wrong value
- Charts not showing price to beat line
- Resolution logic not working

**UI Issues:**
- Labels still showing "STRIKE" instead of "PRICE TO BEAT"
- Missing data in displays

## Quick Test Script
```python
# test_refactor.py - Quick smoke test
from src.core import FingerBlasterCore
from connectors.polymarket import PolymarketConnector

# Test that functions exist
core = FingerBlasterCore()
assert hasattr(core, '_parse_price_to_beat'), "Missing _parse_price_to_beat method"
assert hasattr(core, '_resolve_dynamic_price_to_beat'), "Missing _resolve_dynamic_price_to_beat method"

# Test connector
connector = PolymarketConnector()
assert hasattr(connector, '_extract_price_to_beat'), "Missing _extract_price_to_beat method"

print("✅ All renamed functions exist!")
```

