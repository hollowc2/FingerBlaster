/**
 * TypeScript types for FingerBlaster Web GUI.
 */

// =============================================================================
// Position & Market Data (existing)
// =============================================================================

export interface Position {
  id: string;
  side: 'Up' | 'Down';
  price: number;
  shares: number;
  pnl: number;
  progress: number;
}

export interface MarketData {
  spotPrice: number;
  priceToBeat: number;
  change24h: number;
  itmProb: number;
  basisPoints: number;
  bpsChange: number;
  edge: number;
  zScore: number;
}

// =============================================================================
// API Response Types
// =============================================================================

export interface MarketInfo {
  active: boolean;
  priceToBeat: string | null;
  endDate: string | null;
}

export interface PriceData {
  yesPrice: number;
  noPrice: number;
  bestBid: number;
  bestAsk: number;
  yesSpread: string;
  noSpread: string;
}

export interface AccountData {
  balance: number;
  yesBalance: number;
  noBalance: number;
  selectedSize: number;
  avgEntryYes: number | null;
  avgEntryNo: number | null;
}

export interface CountdownData {
  timeLeft: string;
  urgency: 'normal' | 'watchful' | 'critical';
  secondsRemaining: number;
}

export interface AnalyticsData {
  basisPoints: number | null;
  zScore: number | null;
  sigmaLabel: string | null;
  fairValueYes: number | null;
  fairValueNo: number | null;
  edgeYes: 'undervalued' | 'overvalued' | 'fair' | null;
  edgeNo: 'undervalued' | 'overvalued' | 'fair' | null;
  edgeBpsYes: number | null;
  edgeBpsNo: number | null;
  unrealizedPnl: number | null;
  pnlPercentage: number | null;
  yesAskDepth: number | null;
  noAskDepth: number | null;
  estimatedSlippageYes: number | null;
  estimatedSlippageNo: number | null;
  regimeDirection: string | null;
  regimeStrength: number | null;
  oracleLagMs: number | null;
  timerUrgency: string | null;
}

export interface ChartPoint {
  x: number;
  y: number;
}

export interface InitialState {
  market: MarketInfo;
  prices: PriceData;
  account: AccountData;
  btcPrice: number;
  priorOutcomes: string[];
}

// =============================================================================
// WebSocket Event Types
// =============================================================================

export type WebSocketEvent =
  | { event: 'initial_state'; data: InitialState }
  | { event: 'price_update'; data: PriceData }
  | { event: 'btc_price'; data: { price: number } }
  | { event: 'countdown'; data: CountdownData }
  | { event: 'account_stats'; data: AccountData }
  | { event: 'analytics'; data: AnalyticsData }
  | { event: 'prior_outcomes'; data: { outcomes: string[] } }
  | { event: 'resolution'; data: { resolution: string | null } }
  | { event: 'market_update'; data: { priceToBeat: string; ends: string } }
  | { event: 'btc_chart'; data: { prices: number[]; priceToBeat: number | null } }
  | { event: 'probability_chart'; data: { data: ChartPoint[] } }
  | { event: 'log'; data: { message: string } }
  | { event: 'pong' };

// =============================================================================
// Hook Return Type
// =============================================================================

export interface FingerBlasterState {
  // Connection
  connected: boolean;
  
  // Market
  marketActive: boolean;
  priceToBeat: string;
  endDate: string | null;
  
  // Prices (0-100 scale for display)
  yesPrice: number;
  noPrice: number;
  bestBid: number;
  bestAsk: number;
  yesSpread: string;
  noSpread: string;
  
  // BTC
  btcPrice: number;
  btcHistory: number[];
  
  // Countdown
  timeLeft: string;
  urgency: 'normal' | 'watchful' | 'critical';
  secondsRemaining: number;
  
  // Account
  balance: number;
  yesBalance: number;
  noBalance: number;
  selectedSize: number;
  avgEntryYes: number | null;
  avgEntryNo: number | null;
  
  // Analytics
  analytics: AnalyticsData | null;
  
  // Prior outcomes
  priorOutcomes: string[];
  
  // Resolution
  resolution: string | null;
  
  // Chart data
  probabilityHistory: ChartPoint[];
  
  // Logs
  logs: string[];
}

export interface FingerBlasterActions {
  placeOrder: (side: 'Up' | 'Down', size?: number) => Promise<void>;
  flatten: () => Promise<void>;
  cancelAll: () => Promise<void>;
  sizeUp: () => Promise<void>;
  sizeDown: () => Promise<void>;
  setSize: (size: number) => void;
  discoverMarket: () => Promise<void>;
}

export type UseFingerBlasterReturn = FingerBlasterState & FingerBlasterActions;
