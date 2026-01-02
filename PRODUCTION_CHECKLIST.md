# Production Deployment Checklist

## 🔐 SECURITY - CRITICAL

### Before Trading Real Money:

- [ ] **SWITCH TO HARDWARE WALLET** 🔴 **CRITICAL**
  - Current: Using `.env` file with plaintext private key (DEVELOPMENT ONLY)
  - Production: Must integrate hardware wallet (Ledger/Trezor) or use system keyring
  - Location: `connectors/polymarket.py:167` - Replace `os.getenv("PRIVATE_KEY")`
  - See: `docs/hardware_wallet_integration.md` (to be created)
  - **DO NOT DEPLOY TO PRODUCTION WITHOUT THIS**

- [ ] **Review wallet permissions**
  - Ensure wallet has only necessary funds
  - Consider using a separate "hot wallet" for trading
  - Keep majority of funds in cold storage

- [ ] **Enable transaction limits**
  - Add max trade size limits
  - Add daily volume limits
  - Add circuit breakers for losses

## 🧪 TESTING

- [ ] Test with small amounts first (< $50)
- [ ] Verify all market resolution logic
- [ ] Test emergency stop mechanisms
- [ ] Verify position management works correctly

## 📊 MONITORING

- [ ] Set up balance monitoring alerts
- [ ] Set up error rate monitoring
- [ ] Set up PnL tracking
- [ ] Configure log rotation (logs can get large)

## 🚀 DEPLOYMENT

- [ ] Use separate `.env` for production (never commit!)
- [ ] Set up automated backups of trading logs
- [ ] Document emergency shutdown procedures
- [ ] Test reconnection logic after network failures

## 📝 NOTES

**Current Status**: DEVELOPMENT MODE
- Testing with small amounts
- Using `.env` file for credentials
- Hardware wallets available but not integrated

**When Ready for Production**:
1. Integrate hardware wallet (Ledger/Trezor)
2. Review all checklist items above
3. Start with minimal capital
4. Gradually scale up after proving stability

---

**Last Updated**: 2026-01-02
**Current Mode**: Development/Testing
**Production Ready**: ❌ NO - Hardware wallet integration required
