/**
 * useFingerBlaster - React hook for FingerBlaster API & WebSocket integration.
 * 
 * Provides real-time market data, account state, and trading actions.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import type {
  PriceData,
  AccountData,
  CountdownData,
  AnalyticsData,
  ChartPoint,
  InitialState,
  UseFingerBlasterReturn,
} from '../types';

// Configuration
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';
const API_KEY = import.meta.env.VITE_API_KEY || '';

// Request settings
const RECONNECT_DELAY_MS = 3000;
const REQUEST_TIMEOUT_MS = 10000;
const MAX_RETRIES = 2;
const MAX_LOGS = 100;

// Error callback type
export type ErrorCallback = (message: string) => void;
export type SuccessCallback = (message: string) => void;

// Helper: fetch with timeout and retries
async function fetchWithRetry(
  url: string,
  options: RequestInit = {},
  onError?: ErrorCallback,
  retries = MAX_RETRIES,
): Promise<Response | null> {
  // Add API key header if configured
  const headers = new Headers(options.headers);
  headers.set('Content-Type', 'application/json');
  if (API_KEY) {
    headers.set('X-API-Key', API_KEY);
  }
  
  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    
    try {
      const response = await fetch(url, {
        ...options,
        headers,
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        const errorMsg = errorData.detail || `HTTP ${response.status}`;
        
        // Log 404 errors with more detail
        if (response.status === 404) {
          console.error(`[fetchWithRetry] 404 Not Found for URL: ${url}`);
          console.error(`[fetchWithRetry] Response status: ${response.status}, statusText: ${response.statusText}`);
          onError?.(`Endpoint not found (404): ${url}. Check if the server is running and the route exists.`);
          return null;
        }
        
        // Rate limit error - don't retry
        if (response.status === 429) {
          onError?.('Rate limit exceeded. Please wait.');
          return null;
        }
        
        // Auth error - don't retry
        if (response.status === 401 || response.status === 403) {
          onError?.('Authentication failed');
          return null;
        }
        
        throw new Error(errorMsg);
      }
      
      return response;
    } catch (e) {
      clearTimeout(timeoutId);
      
      if (e instanceof Error) {
        if (e.name === 'AbortError') {
          console.warn(`Request timeout (attempt ${attempt + 1}/${retries + 1})`);
        } else {
          console.warn(`Request failed (attempt ${attempt + 1}/${retries + 1}):`, e.message);
        }
        
        // Last attempt failed
        if (attempt === retries) {
          onError?.(e.message || 'Request failed');
          return null;
        }
      }
      
      // Wait before retry (exponential backoff)
      await new Promise(resolve => setTimeout(resolve, 500 * Math.pow(2, attempt)));
    }
  }
  
  return null;
}

interface UseFingerBlasterOptions {
  onError?: ErrorCallback;
  onSuccess?: SuccessCallback;
}

export function useFingerBlaster(options: UseFingerBlasterOptions = {}): UseFingerBlasterReturn {
  const { onError, onSuccess } = options;
  
  // Connection state
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Market state
  const [marketActive, setMarketActive] = useState(false);
  const [priceToBeat, setPriceToBeat] = useState('--');
  const [endDate, setEndDate] = useState<string | null>(null);

  // Price state (stored as 0-1, converted to 0-100 for display)
  const [prices, setPrices] = useState<PriceData>({
    yesPrice: 0.5,
    noPrice: 0.5,
    bestBid: 0.5,
    bestAsk: 0.5,
    yesSpread: '-- / --',
    noSpread: '-- / --',
  });

  // BTC state
  const [btcPrice, setBtcPrice] = useState(0);
  const [btcHistory, setBtcHistory] = useState<number[]>([]);

  // Countdown state
  const [countdown, setCountdown] = useState<CountdownData>({
    timeLeft: '--:--',
    urgency: 'normal',
    secondsRemaining: 0,
  });

  // Account state
  const [account, setAccount] = useState<AccountData>({
    balance: 0,
    yesBalance: 0,
    noBalance: 0,
    selectedSize: 1,
    avgEntryYes: null,
    avgEntryNo: null,
  });

  // Local size state for immediate UI feedback
  const [localSize, setLocalSize] = useState(1);

  // Analytics state
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);

  // Prior outcomes
  const [priorOutcomes, setPriorOutcomes] = useState<string[]>([]);

  // Resolution
  const [resolution, setResolution] = useState<string | null>(null);

  // Chart data
  const [probabilityHistory, setProbabilityHistory] = useState<ChartPoint[]>([]);

  // Logs
  const [logs, setLogs] = useState<string[]>([]);

  // Process initial state from WebSocket
  const processInitialState = useCallback((data: InitialState) => {
    setMarketActive(data.market.active);
    // Handle price to beat - check for null, undefined, 'N/A', etc.
    const strikeVal = data.market.priceToBeat;
    if (strikeVal && strikeVal !== 'N/A' && strikeVal !== 'None' && strikeVal !== '') {
      setPriceToBeat(strikeVal);
      console.log('Price to beat set from initial state:', strikeVal);
    } else {
      setPriceToBeat('--');
      console.log('Price to beat not available. Received:', strikeVal, 'Market active:', data.market.active);
    }
    setEndDate(data.market.endDate);
    setPrices(data.prices);
    setAccount(data.account);
    setLocalSize(data.account.selectedSize);
    setBtcPrice(data.btcPrice);
    setPriorOutcomes(data.priorOutcomes);
  }, []);

  // Use ref for localSize to avoid dependency issues
  const localSizeRef = useRef(localSize);
  useEffect(() => {
    localSizeRef.current = localSize;
  }, [localSize]);

  // Ref to store the latest message handler to avoid reconnection loops
  const handleMessageRef = useRef<((event: MessageEvent) => void) | null>(null);

  // WebSocket message handler - use ref to avoid reconnection loops
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const msg = JSON.parse(event.data);
      
      switch (msg.event) {
        case 'initial_state':
          processInitialState(msg.data);
          break;
          
        case 'price_update':
          setPrices(msg.data);
          break;
          
        case 'btc_price':
          setBtcPrice(msg.data.price);
          break;
          
        case 'countdown':
          setCountdown(msg.data);
          break;
          
        case 'account_stats':
          setAccount(msg.data);
          // Sync local size with server if significantly different
          if (Math.abs(msg.data.selectedSize - localSizeRef.current) > 0.01) {
            setLocalSize(msg.data.selectedSize);
          }
          break;
          
        case 'analytics':
          setAnalytics(msg.data);
          break;
          
        case 'prior_outcomes':
          setPriorOutcomes(msg.data.outcomes);
          break;
          
        case 'resolution':
          setResolution(msg.data.resolution);
          break;
          
        case 'market_update':
          const strikeVal = msg.data.priceToBeat;
          if (strikeVal && strikeVal !== 'N/A' && strikeVal !== 'None' && strikeVal !== '') {
            setPriceToBeat(strikeVal);
            console.log('Price to beat updated from market_update:', strikeVal);
          } else {
            console.log('Market update received but price to beat invalid:', strikeVal);
          }
          setMarketActive(true);
          break;
          
        case 'btc_chart':
          setBtcHistory(msg.data.prices || []);
          break;
          
        case 'probability_chart':
          setProbabilityHistory(msg.data.data || []);
          break;
          
        case 'log':
          setLogs(prev => {
            const newLogs = [...prev, msg.data.message];
            return newLogs.slice(-MAX_LOGS);
          });
          break;
          
        case 'pong':
          // Heartbeat response, ignore
          break;
      }
    } catch (e) {
      console.error('WebSocket message parse error:', e);
    }
  }, [processInitialState]);

  // Update message handler ref when it changes - initialize immediately
  handleMessageRef.current = handleMessage;

  // Connect to WebSocket - stable reference to avoid reconnection loops
  const connect = useCallback(() => {
    // Don't reconnect if already connected or connecting
    if (wsRef.current?.readyState === WebSocket.OPEN || 
        wsRef.current?.readyState === WebSocket.CONNECTING) {
      return;
    }

    // Clean up any existing connection
    if (wsRef.current) {
      try {
        wsRef.current.onclose = null;
        wsRef.current.onerror = null;
        wsRef.current.close();
      } catch (e) {
        // Ignore errors during cleanup
      }
    }

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      console.log('WebSocket connected');
    };

    ws.onclose = (event) => {
      setConnected(false);
      // Only log and reconnect if not a normal closure or if we're not already reconnecting
      if (event.code !== 1000 && event.code !== 1001) {
        console.log('WebSocket disconnected, reconnecting...', event.code, event.reason);
        
        // Schedule reconnect only if not already scheduled
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
        }
        reconnectTimeoutRef.current = setTimeout(() => {
          reconnectTimeoutRef.current = null;
          connect();
        }, RECONNECT_DELAY_MS);
      } else {
        console.log('WebSocket closed normally');
      }
    };

    ws.onerror = (error) => {
      // Log error but don't immediately close - let onclose handle reconnection
      console.error('WebSocket error:', error);
      // Don't call ws.close() here - let the browser handle it
    };

    ws.onmessage = (event) => {
      // Use the ref to call the latest handler
      if (handleMessageRef.current) {
        handleMessageRef.current(event);
      }
    };
  }, []); // Empty deps - connect should be stable

  // Initialize WebSocket connection - only run once on mount
  useEffect(() => {
    connect();

    return () => {
      // Clean up on unmount
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      if (wsRef.current) {
        try {
          wsRef.current.onclose = null;
          wsRef.current.onerror = null;
          wsRef.current.close(1000, 'Component unmounting');
        } catch (e) {
          // Ignore errors during cleanup
        }
        wsRef.current = null;
      }
    };
  }, []); // Empty deps - only run once on mount

  // API Actions
  const placeOrder = useCallback(async (side: 'YES' | 'NO', size?: number) => {
    const response = await fetchWithRetry(
      `${API_BASE}/api/order`,
      {
        method: 'POST',
        body: JSON.stringify({ side, size: size ?? localSize }),
      },
      onError,
    );
    
    if (response) {
      onSuccess?.(`${side} order submitted`);
    }
  }, [localSize, onError, onSuccess]);

  const flatten = useCallback(async () => {
    const response = await fetchWithRetry(
      `${API_BASE}/api/flatten`,
      { method: 'POST' },
      onError,
    );
    
    if (response) {
      onSuccess?.('Flattening positions...');
    }
  }, [onError, onSuccess]);

  const cancelAll = useCallback(async () => {
    const response = await fetchWithRetry(
      `${API_BASE}/api/cancel`,
      { method: 'POST' },
      onError,
    );
    
    if (response) {
      onSuccess?.('Cancelling all orders...');
    }
  }, [onError, onSuccess]);

  const sizeUp = useCallback(async () => {
    // Optimistic update
    setLocalSize(prev => prev + 1);
    
    const response = await fetchWithRetry(
      `${API_BASE}/api/size`,
      {
        method: 'POST',
        body: JSON.stringify({ action: 'up' }),
      },
      onError,
    );
    
    if (response) {
      const data = await response.json();
      setLocalSize(data.size);
    }
  }, [onError]);

  const sizeDown = useCallback(async () => {
    // Optimistic update (minimum $1)
    setLocalSize(prev => Math.max(1, prev - 1));
    
    const response = await fetchWithRetry(
      `${API_BASE}/api/size`,
      {
        method: 'POST',
        body: JSON.stringify({ action: 'down' }),
      },
      onError,
    );
    
    if (response) {
      const data = await response.json();
      setLocalSize(data.size);
    }
  }, [onError]);

  const setSize = useCallback((size: number) => {
    setLocalSize(Math.max(1, size));
  }, []);

  const discoverMarket = useCallback(async () => {
    const url = `${API_BASE}/api/discover-market`;
    console.log(`[discoverMarket] Calling: ${url}`);
    
    const response = await fetchWithRetry(
      url,
      { method: 'POST' },
      (error) => {
        console.error(`[discoverMarket] Error: ${error}`);
        onError?.(error || 'Failed to discover market');
      },
    );
    
    if (response) {
      console.log(`[discoverMarket] Success: ${response.status}`);
      onSuccess?.('Market discovery triggered');
    } else {
      console.warn(`[discoverMarket] No response received`);
    }
  }, [onError, onSuccess]);

  // Return combined state and actions
  return {
    // Connection
    connected,
    
    // Market
    marketActive,
    priceToBeat,
    endDate,
    
    // Prices (convert to 0-100 scale for display)
    yesPrice: Math.round(prices.yesPrice * 100),
    noPrice: Math.round(prices.noPrice * 100),
    bestBid: prices.bestBid,
    bestAsk: prices.bestAsk,
    yesSpread: prices.yesSpread,
    noSpread: prices.noSpread,
    
    // BTC
    btcPrice,
    btcHistory,
    
    // Countdown
    timeLeft: countdown.timeLeft,
    urgency: countdown.urgency,
    secondsRemaining: countdown.secondsRemaining,
    
    // Account
    balance: account.balance,
    yesBalance: account.yesBalance,
    noBalance: account.noBalance,
    selectedSize: localSize,
    avgEntryYes: account.avgEntryYes,
    avgEntryNo: account.avgEntryNo,
    
    // Analytics
    analytics,
    
    // Prior outcomes
    priorOutcomes,
    
    // Resolution
    resolution,
    
    // Chart data
    probabilityHistory,
    
    // Logs
    logs,
    
    // Actions
    placeOrder,
    flatten,
    cancelAll,
    sizeUp,
    sizeDown,
    setSize,
    discoverMarket,
  };
}

