import React, { useMemo, useRef, useEffect, useState } from 'react';
import StatCard from './components/StatCard';
import OrderDepth from './components/OrderDepth';
import ExecutionControls from './components/ExecutionControls';
import { ToastProvider, useToast } from './components/Toast';
import { useFingerBlaster } from './hooks/useFingerBlaster';

// Inner component that uses the toast context
const AppContent: React.FC = () => {
  const toast = useToast();
  const fb = useFingerBlaster({
    onError: toast.error,
    onSuccess: toast.success,
  });

  // Track BTC price direction for flashing arrow
  const prevBtcPriceRef = useRef<number | null>(null);
  const [btcPriceDirection, setBtcPriceDirection] = useState<'up' | 'down' | null>(null);
  const [btcPriceFlashKey, setBtcPriceFlashKey] = useState(0);

  useEffect(() => {
    if (prevBtcPriceRef.current !== null && fb.btcPrice !== prevBtcPriceRef.current) {
      const direction = fb.btcPrice > prevBtcPriceRef.current ? 'up' : 'down';
      setBtcPriceDirection(direction);
      setBtcPriceFlashKey(prev => prev + 1);
    }
    prevBtcPriceRef.current = fb.btcPrice;
  }, [fb.btcPrice]);

  // Generate SVG path for the BTC price chart
  const { chartPath, fillPath, currentPrice, minPrice, maxPrice } = useMemo(() => {
    const prices = fb.btcHistory.length > 0 ? fb.btcHistory : [fb.btcPrice || 0];
    if (prices.length < 2 || prices.every(p => p === 0)) {
      return { chartPath: '', fillPath: '', currentPrice: fb.btcPrice, minPrice: 0, maxPrice: 1 };
    }
    
    const min = Math.min(...prices) - 10;
    const max = Math.max(...prices) + 10;
    const range = max - min || 1;
    
    const points = prices.map((val, idx) => {
      const x = (idx / (prices.length - 1)) * 100;
      const y = 100 - ((val - min) / range) * 100;
      return `${x},${y}`;
    });
    
    const path = `M ${points.join(" L ")}`;
    const fill = `${path} L 100,100 L 0,100 Z`;
    
    return { 
      chartPath: path, 
      fillPath: fill, 
      currentPrice: prices[prices.length - 1],
      minPrice: min,
      maxPrice: max,
    };
  }, [fb.btcHistory, fb.btcPrice]);

  // Parse price to beat for display
  const strikeValue = useMemo(() => {
    const parsed = parseFloat(fb.priceToBeat.replace(/,/g, ''));
    return isNaN(parsed) ? null : parsed;
  }, [fb.priceToBeat]);

  // Calculate BTC delta from price to beat
  const btcDelta = useMemo(() => {
    if (!strikeValue || !fb.btcPrice) return null;
    return fb.btcPrice - strikeValue;
  }, [fb.btcPrice, strikeValue]);

  // Calculate price to beat line Y position in chart coordinates
  const strikeLineY = useMemo(() => {
    if (!strikeValue || minPrice === maxPrice) return null;
    const y = 100 - ((strikeValue - minPrice) / (maxPrice - minPrice || 1)) * 100;
    return Math.max(0, Math.min(100, y));
  }, [strikeValue, minPrice, maxPrice]);

  // Calculate Z-score color gradient
  const zScoreColor = useMemo(() => {
    if (fb.analytics?.zScore == null) return 'text-white';
    const zScore = fb.analytics.zScore;
    const absZ = Math.abs(zScore);
    // Clamp to 0-3 range for intensity
    const intensity = Math.min(absZ / 3, 1);
    
    if (zScore > 0) {
      // Green gradient: brighter as it moves away from zero
      const opacity = 0.5 + (intensity * 0.5); // 0.5 to 1.0
      return { color: `rgba(6, 249, 87, ${opacity})`, textClass: 'text-primary' };
    } else {
      // Red gradient: brighter as it moves away from zero
      const opacity = 0.5 + (intensity * 0.5); // 0.5 to 1.0
      return { color: `rgba(255, 59, 48, ${opacity})`, textClass: 'text-accent-red' };
    }
  }, [fb.analytics?.zScore]);

  // Format prior outcomes as arrows
  const priorOutcomesDisplay = useMemo(() => {
    return fb.priorOutcomes.slice(0, 10).map((outcome, idx) => (
      <span 
        key={idx} 
        className={outcome === 'Up' ? 'text-primary' : 'text-accent-red'}
      >
        {outcome === 'Up' ? '▲' : '▼'}
      </span>
    ));
  }, [fb.priorOutcomes]);

  return (
    <div 
      className="relative flex flex-col min-h-screen w-full bg-background-dark selection:bg-primary selection:text-black"
      style={{ backgroundColor: '#050a07', minHeight: '100vh' }}
    >
      {/* Resolution Overlay */}
      {fb.resolution && (
        <div className={`fixed inset-0 z-[100] flex items-center justify-center transition-opacity duration-300 ${
          fb.resolution === 'Up' ? 'bg-primary' : 'bg-accent-red'
        }`}>
          <div className="text-center">
            <h1 className={`text-9xl font-black ${fb.resolution === 'Up' ? 'text-black' : 'text-white'}`}>
              {fb.resolution}
            </h1>
            <p className={`text-2xl font-bold mt-4 ${fb.resolution === 'Up' ? 'text-black/70' : 'text-white/70'}`}>
              MARKET RESOLVED
            </p>
          </div>
        </div>
      )}

      {/* Header */}
      <header className="sticky top-0 z-50 flex items-center justify-between border-b border-primary/20 bg-surface-darker/90 backdrop-blur-md px-6 py-3">
        <div className="flex items-center gap-4">
          <div className="flex items-center justify-center size-8 rounded bg-primary/10 border border-primary text-primary">
            <span className="material-symbols-outlined text-xl">terminal</span>
          </div>
          <div>
            <h2 className="text-white text-lg font-bold tracking-tighter leading-none">FINGER BLASTER</h2>
            <div className="flex items-center gap-2 text-[10px] tracking-widest font-mono mt-0.5">
              <span className={`inline-block size-1.5 rounded-full ${fb.connected ? 'bg-primary animate-pulse' : 'bg-accent-red'}`}></span>
              <span className={fb.connected ? 'text-primary/70' : 'text-accent-red/70'}>
                {fb.connected ? 'SYSTEM ONLINE' : 'DISCONNECTED'}
              </span>
            </div>
          </div>
        </div>

        <div className="hidden md:flex items-center gap-6 border-x border-primary/10 px-6 h-full">
          {[
            { label: 'Oracle Lag', value: fb.analytics?.oracleLagMs ? `${fb.analytics?.oracleLagMs}ms` : 'SYNC', color: (fb.analytics?.oracleLagMs ?? 0) > 500 ? 'text-accent-orange' : 'text-primary' },
            { label: 'Regime', value: fb.analytics?.regimeDirection || 'NEUTRAL', color: fb.analytics?.regimeDirection === 'BULLISH' ? 'text-primary' : fb.analytics?.regimeDirection === 'BEARISH' ? 'text-accent-red' : 'text-accent-orange' },
            { label: 'Network', value: 'POLYGON', color: 'text-white' },
          ].map((item, idx) => (
            <React.Fragment key={item.label}>
              <div className="flex flex-col items-center">
                <span className="text-[10px] text-gray-400 uppercase tracking-widest">{item.label}</span>
                <span className={`text-sm font-bold font-mono ${item.color}`}>{item.value}</span>
              </div>
              {idx < 2 && <div className="w-px h-6 bg-primary/10" />}
            </React.Fragment>
          ))}
        </div>

        <div className="flex gap-3">
          <button className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded border border-primary/30 bg-primary/5 hover:bg-primary/10 transition-colors text-xs font-bold text-primary uppercase tracking-wider">
            <span className="material-symbols-outlined text-sm">account_balance_wallet</span>
            <span>${fb.balance.toFixed(2)}</span>
          </button>
        </div>
      </header>

      {/* Main Dashboard */}
      <main className="flex-1 p-4 md:p-6 grid grid-cols-12 gap-4 md:gap-6 max-w-[1600px] mx-auto w-full">
        
        {/* Left Stats Column */}
        <aside className="col-span-12 lg:col-span-3 flex flex-col gap-4">
          <div className="flex items-center gap-2 pb-2 border-b border-primary/20">
            <span className="material-symbols-outlined text-primary text-sm">data_usage</span>
            <h3 className="text-sm font-bold tracking-widest text-white uppercase">Data Stream // BTC-15M</h3>
          </div>
          
          <div className="grid grid-cols-2 lg:grid-cols-1 gap-3">
            <StatCard 
              label="BTC Price" 
              value={fb.btcPrice ? `$${fb.btcPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '--'}
              subValue={btcDelta !== null ? `${btcDelta >= 0 ? '+' : ''}${btcDelta.toFixed(0)} from price to beat` : ''}
              subValueColor={btcDelta !== null && btcDelta >= 0 ? 'text-primary' : 'text-accent-red'}
              trend={btcDelta !== null && btcDelta >= 0 ? 'up' : 'down'}
              icon="show_chart"
              priceDirection={btcPriceDirection}
              flashKey={btcPriceFlashKey}
            />
            <div className="bg-surface-dark border border-white/5 p-4 rounded-lg">
              <p className="text-gray-400 text-xs font-medium uppercase tracking-wider mb-1">Price to Beat</p>
              <p className="text-2xl font-bold text-white tracking-tight">
                {fb.priceToBeat && fb.priceToBeat !== '--' && fb.priceToBeat !== 'N/A' ? `$${fb.priceToBeat}` : '--'}
              </p>
              <div className="w-full bg-white/10 h-1 mt-2 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-blue-500" 
                  style={{ width: `${fb.yesPrice}%` }}
                ></div>
              </div>
              <p className="text-xs text-gray-500 mt-1 text-right">ITM Probability: {fb.yesPrice}%</p>
            </div>
            <StatCard 
              label="Basis Points (BPS)" 
              value={fb.analytics?.basisPoints != null ? fb.analytics?.basisPoints.toFixed(0) : '--'}
              trend={(fb.analytics?.basisPoints ?? 0) >= 0 ? 'up' : 'down'}
              valueColor={(fb.analytics?.basisPoints ?? 0) >= 0 ? 'text-primary' : 'text-accent-red'}
            />
            <StatCard 
              label="Edge Detected" 
              value={fb.analytics?.edgeYes ? `${fb.analytics.edgeYes === 'undervalued' ? '+' : '-'}${Math.abs(fb.analytics?.edgeBpsYes || 0).toFixed(0)}bps` : '--'}
              subValue={fb.analytics?.edgeYes === 'undervalued' ? 'BUY Signal' : fb.analytics?.edgeYes === 'overvalued' ? 'SELL Signal' : 'Fair Value'}
              subValueColor={fb.analytics?.edgeYes === 'undervalued' ? 'text-primary' : fb.analytics?.edgeYes === 'overvalued' ? 'text-accent-red' : 'text-gray-400'}
              highlight={fb.analytics?.edgeYes === 'undervalued'}
              valueColor={fb.analytics?.edgeBpsYes != null && fb.analytics.edgeBpsYes >= 0 ? 'text-primary' : 'text-accent-red'}
              trend={fb.analytics?.edgeBpsYes != null && fb.analytics.edgeBpsYes >= 0 ? 'up' : 'down'}
              flashBlock={fb.analytics?.edgeYes === 'undervalued'}
            />
            <StatCard 
              label="Z-Score" 
              value={fb.analytics?.zScore?.toFixed(2) || '--'}
              valueColor={zScoreColor.textClass}
              customValueStyle={{ color: zScoreColor.color }}
              valueIcon="σ"
            />
            {fb.analytics?.fairValueYes != null && (
              <StatCard 
                label="Fair Value" 
                value={`${(fb.analytics.fairValueYes * 100).toFixed(1)}%`}
                valueColor={fb.yesPrice > (fb.analytics.fairValueYes * 100) ? 'text-accent-red' : 'text-primary'}
              />
            )}
          </div>
        </aside>

        {/* Center Panel */}
        <section className="col-span-12 lg:col-span-6 flex flex-col gap-6">
          {/* Market Expiry Timer */}
          <div className={`bg-surface-dark border rounded-xl p-6 flex flex-col items-center justify-center relative overflow-hidden min-h-[160px] transition-colors ${
            fb.urgency === 'critical' ? 'border-accent-red/50 bg-accent-red/5' : 
            fb.urgency === 'watchful' ? 'border-accent-orange/50' : 
            'border-white/10'
          }`}>
            <div className="absolute inset-0 opacity-10" style={{ backgroundImage: `linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)`, backgroundSize: '20px 20px' }}></div>
            <h4 className="text-primary/80 font-mono text-sm uppercase tracking-[0.2em] mb-2 z-10">Market Expiry Remaining</h4>
            <div className={`text-6xl md:text-8xl font-bold tracking-tighter font-mono z-10 tabular-nums drop-shadow-[0_0_15px_rgba(255,255,255,0.3)] ${
              fb.urgency === 'critical' ? 'text-accent-red animate-pulse' : 
              fb.urgency === 'watchful' ? 'text-accent-orange' : 
              'text-white'
            }`}>
              {fb.timeLeft}
            </div>
            <div className="flex gap-8 mt-4 z-10">
              <div className="flex items-center gap-2">
                <div className={`size-2 rounded-full ${fb.marketActive ? 'bg-primary shadow-[0_0_8px_#06f957]' : 'bg-gray-500'}`}></div>
                <span className={`text-xs font-bold uppercase tracking-widest ${fb.marketActive ? 'text-primary' : 'text-gray-500'}`}>
                  {fb.marketActive ? 'Live' : 'No Market'}
                </span>
                {!fb.marketActive && (
                  <button
                    onClick={fb.discoverMarket}
                    className="ml-2 px-2 py-1 text-[10px] bg-primary/20 border border-primary/50 text-primary hover:bg-primary/30 rounded uppercase tracking-wider font-bold transition-colors"
                    title="Search for active market"
                  >
                    Search
                  </button>
                )}
              </div>
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-gray-500 text-sm">history</span>
                <span className="text-xs text-gray-400 font-bold uppercase tracking-widest flex gap-0.5">
                  {priorOutcomesDisplay.length > 0 ? priorOutcomesDisplay : '---'}
                </span>
              </div>
            </div>
          </div>

          {/* Chart View */}
          <div className="bg-surface-dark border border-white/10 rounded-xl p-1 flex-1 min-h-[150px] max-h-[150px] relative flex flex-col overflow-hidden">
            <div className="flex justify-between items-center px-4 py-3 border-b border-white/5 bg-surface-dark/50 z-20">
              <div className="flex gap-4">
                <span className="text-xs font-bold text-white bg-primary/20 border border-primary/50 px-2 py-1 rounded cursor-pointer">LIVE</span>
              </div>
              <div className="text-xs text-primary font-mono font-bold flex items-center gap-2">
                <span className={`size-1.5 rounded-full ${fb.connected ? 'bg-primary animate-ping' : 'bg-gray-500'}`}></span>
                SPOT: ${fb.btcPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </div>
            </div>
            
            <div className="relative w-full h-full flex items-end overflow-hidden px-4 pb-4 mt-2">
              <div className="absolute top-1/2 w-full border-t border-dashed border-white/10 z-0"></div> 
              
              <svg className="w-full h-full z-10" preserveAspectRatio="none" viewBox="0 0 100 100">
                <defs>
                  <linearGradient id="chartGradientGreen" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0%" stopColor="#06f957" stopOpacity="0.4"></stop>
                    <stop offset="100%" stopColor="#06f957" stopOpacity="0"></stop>
                  </linearGradient>
                  <linearGradient id="chartGradientRed" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0%" stopColor="#FF0000" stopOpacity="0.4"></stop>
                    <stop offset="100%" stopColor="#FF0000" stopOpacity="0"></stop>
                  </linearGradient>
                </defs>
                {fillPath && (
                  <path 
                    d={fillPath} 
                    fill={strikeValue && fb.btcPrice && fb.btcPrice < strikeValue ? "url(#chartGradientRed)" : "url(#chartGradientGreen)"} 
                    vectorEffect="non-scaling-stroke"
                  />
                )}
                {chartPath && (
                  <path 
                    d={chartPath} 
                    fill="none" 
                    stroke={strikeValue && fb.btcPrice && fb.btcPrice < strikeValue ? "#FF0000" : "#06f957"} 
                    strokeWidth="3" 
                    vectorEffect="non-scaling-stroke" 
                    strokeLinecap="round" 
                    strokeLinejoin="round"
                  />
                )}
                
                {/* Price to Beat Line - Always Bright Yellow Horizontal Line */}
                {strikeLineY !== null && (
                  <line 
                    x1="0" 
                    y1={strikeLineY}
                    x2="100" 
                    y2={strikeLineY}
                    stroke="#FFFF00" 
                    strokeWidth="1" 
                    strokeOpacity="1"
                    style={{ filter: 'drop-shadow(0 0 2px rgba(255, 255, 0, 0.8))' }}
                  />
                )}
                
                {/* Dynamic Price Marker */}
                {chartPath && (
                  <circle 
                    cx="100" 
                    cy={100 - ((currentPrice - minPrice) / (maxPrice - minPrice || 1)) * 100} 
                    r="1.5" 
                    fill={strikeValue && fb.btcPrice && fb.btcPrice < strikeValue ? "#FF0000" : "#06f957"} 
                    className="animate-pulse shadow-glow"
                  ></circle>
                )}
              </svg>

              {strikeValue && (
                <div className="absolute top-1/2 right-4 text-[10px] text-white/40 -mt-3 bg-surface-dark px-1 font-bold z-20 border border-white/10 rounded">
                  PRICE TO BEAT: ${strikeValue.toLocaleString()}
                </div>
              )}
            </div>
          </div>

          <ExecutionControls 
            yesPrice={fb.yesPrice} 
            noPrice={fb.noPrice}
            selectedSize={fb.selectedSize}
            onPlaceOrder={fb.placeOrder}
            onSizeUp={fb.sizeUp}
            onSizeDown={fb.sizeDown}
            onSetSize={fb.setSize}
            disabled={!fb.connected || !fb.marketActive || fb.urgency === 'critical'}
            disabledReason={
              !fb.connected 
                ? 'WebSocket disconnected' 
                : !fb.marketActive 
                  ? 'No active market' 
                  : fb.urgency === 'critical'
                    ? 'Market expiring soon'
                    : undefined
            }
          />
        </section>

        {/* Right Portfolio Column */}
        <aside className="col-span-12 lg:col-span-3 flex flex-col gap-4">
          {/* Unrealized PnL */}
          <div className="bg-surface-dark border border-white/10 rounded-lg p-5">
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs text-gray-400 uppercase tracking-widest font-bold">Unrealized PnL</span>
              <span className={`size-2 rounded-full ${fb.connected ? 'bg-primary animate-pulse' : 'bg-gray-500'}`}></span>
            </div>
            <div className={`text-4xl font-bold tracking-tight mb-1 font-mono ${
              (fb.analytics?.unrealizedPnl ?? 0) >= 0 ? 'text-primary' : 'text-accent-red'
            }`}>
              {(fb.analytics?.unrealizedPnl ?? 0) >= 0 ? '+' : ''}${(fb.analytics?.unrealizedPnl ?? 0).toFixed(2)}
            </div>
            <div className={`flex items-center gap-2 text-xs ${
              (fb.analytics?.pnlPercentage ?? 0) >= 0 ? 'text-primary/80' : 'text-accent-red/80'
            }`}>
              <span className="material-symbols-outlined text-[16px]">
                {(fb.analytics?.pnlPercentage ?? 0) >= 0 ? 'trending_up' : 'trending_down'}
              </span>
              <span>{(fb.analytics?.pnlPercentage ?? 0) >= 0 ? '+' : ''}{(fb.analytics?.pnlPercentage ?? 0).toFixed(1)}% Today</span>
            </div>
          </div>

          {/* Open Positions List */}
          <div className="bg-surface-dark border border-white/5 rounded-lg flex-1 flex flex-col overflow-hidden max-h-[250px]">
            <div className="p-4 border-b border-white/5 flex items-center justify-between">
              <h3 className="text-sm font-bold text-white uppercase tracking-widest">Open Positions</h3>
              <span className="bg-white/10 text-white text-[10px] px-1.5 py-0.5 rounded font-bold">
                {(fb.yesBalance > 0.1 ? 1 : 0) + (fb.noBalance > 0.1 ? 1 : 0)}
              </span>
            </div>
            <div className="flex-1 overflow-y-auto custom-scrollbar">
              {fb.yesBalance > 0.1 && (
                <div className="p-4 border-b border-white/5 hover:bg-white/5 transition-colors cursor-pointer group">
                  <div className="flex justify-between items-start mb-2">
                    <span className="font-bold text-sm text-primary">
                      Up <span className="text-gray-400 font-normal">@ {fb.avgEntryYes ? (fb.avgEntryYes * 100).toFixed(0) : '--'}¢</span>
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">{fb.yesBalance.toLocaleString()} Shares</span>
                    <span className={`text-sm font-bold font-mono ${
                      fb.yesPrice > (fb.avgEntryYes || 0) * 100 ? 'text-primary' : 'text-accent-red'
                    }`}>
                      {fb.yesPrice > (fb.avgEntryYes || 0) * 100 ? '+' : ''}{((fb.yesPrice / 100 - (fb.avgEntryYes || 0)) * fb.yesBalance).toFixed(2)}
                    </span>
                  </div>
                  <div className="mt-2 w-full bg-white/5 h-1 rounded-full overflow-hidden">
                    <div className="bg-primary h-full" style={{ width: `${fb.yesPrice}%` }}></div>
                  </div>
                </div>
              )}
              {fb.noBalance > 0.1 && (
                <div className="p-4 border-b border-white/5 hover:bg-white/5 transition-colors cursor-pointer group">
                  <div className="flex justify-between items-start mb-2">
                    <span className="font-bold text-sm text-accent-red">
                      Down <span className="text-gray-400 font-normal">@ {fb.avgEntryNo ? (fb.avgEntryNo * 100).toFixed(0) : '--'}¢</span>
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">{fb.noBalance.toLocaleString()} Shares</span>
                    <span className={`text-sm font-bold font-mono ${
                      fb.noPrice > (fb.avgEntryNo || 0) * 100 ? 'text-primary' : 'text-accent-red'
                    }`}>
                      {fb.noPrice > (fb.avgEntryNo || 0) * 100 ? '+' : ''}{((fb.noPrice / 100 - (fb.avgEntryNo || 0)) * fb.noBalance).toFixed(2)}
                    </span>
                  </div>
                  <div className="mt-2 w-full bg-white/5 h-1 rounded-full overflow-hidden">
                    <div className="bg-accent-red h-full" style={{ width: `${fb.noPrice}%` }}></div>
                  </div>
                </div>
              )}
              {fb.yesBalance <= 0.1 && fb.noBalance <= 0.1 && (
                <div className="p-4 text-center text-gray-500 text-sm">
                  No open positions
                </div>
              )}
            </div>
          </div>

          <OrderDepth 
            yesAskDepth={fb.analytics?.yesAskDepth ?? 0}
            noAskDepth={fb.analytics?.noAskDepth ?? 0}
          />

          <button 
            onClick={fb.flatten}
            disabled={!fb.connected || (fb.yesBalance === 0 && fb.noBalance === 0)}
            className="w-full py-3 bg-red-900/20 border border-red-900/50 text-red-500 hover:bg-red-900/40 hover:text-red-400 rounded-lg text-xs font-bold uppercase tracking-widest transition-all flex items-center justify-center gap-2 group shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <span className="material-symbols-outlined text-sm group-hover:animate-ping">warning</span>
            PANIC CLOSE ALL
          </button>
        </aside>
      </main>
    </div>
  );
};

// Main App component wrapped with ToastProvider
const App: React.FC = () => (
  <ToastProvider>
    <AppContent />
  </ToastProvider>
);

export default App;

