import React, { useState, useCallback } from 'react';

interface ExecutionControlsProps {
  yesPrice: number;
  noPrice: number;
  selectedSize: number;
  onPlaceOrder: (side: 'YES' | 'NO', size?: number) => Promise<void>;
  onSizeUp: () => Promise<void>;
  onSizeDown: () => Promise<void>;
  onSetSize: (size: number) => void;
  disabled?: boolean;
  disabledReason?: string;
}

const ExecutionControls: React.FC<ExecutionControlsProps> = ({ 
  yesPrice, 
  noPrice,
  selectedSize,
  onPlaceOrder,
  onSizeUp,
  onSizeDown,
  onSetSize,
  disabled = false,
  disabledReason,
}) => {
  const [slippage, setSlippage] = useState('0.5%');
  const [isSubmitting, setIsSubmitting] = useState<'YES' | 'NO' | null>(null);

  // Format size for display
  const formatSize = (size: number): string => {
    return size.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  };

  // Handle size input change
  const handleSizeChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value.replace(/[^0-9.]/g, '');
    const parsed = parseFloat(value);
    if (!isNaN(parsed) && parsed >= 0) {
      onSetSize(parsed);
    } else if (value === '' || value === '.') {
      onSetSize(0);
    }
  }, [onSetSize]);

  // Handle order placement with loading state
  const handleOrder = useCallback(async (side: 'YES' | 'NO') => {
    if (disabled || isSubmitting) return;
    
    setIsSubmitting(side);
    try {
      await onPlaceOrder(side);
    } finally {
      setIsSubmitting(null);
    }
  }, [disabled, isSubmitting, onPlaceOrder]);

  // Percentage buttons
  const handlePercentage = useCallback((percent: number) => {
    // This would need balance info to calculate - for now just multiply
    const newSize = Math.max(1, Math.round(selectedSize * percent / 100));
    onSetSize(newSize);
  }, [selectedSize, onSetSize]);

  return (
    <div className="bg-surface-darker border-t-2 border-primary/20 p-6 rounded-xl shadow-2xl relative overflow-hidden">
      {/* Corner Accents */}
      <div className="absolute top-0 left-0 size-4 border-l-2 border-t-2 border-primary"></div>
      <div className="absolute top-0 right-0 size-4 border-r-2 border-t-2 border-primary"></div>
      <div className="absolute bottom-0 left-0 size-4 border-l-2 border-b-2 border-primary"></div>
      <div className="absolute bottom-0 right-0 size-4 border-r-2 border-b-2 border-primary"></div>

      <div className="flex items-center justify-between mb-4">
        <h3 className="text-white font-bold tracking-widest uppercase flex items-center gap-2">
          <span className="material-symbols-outlined text-primary">bolt</span>
          Execution Controls
        </h3>
        <div className="flex items-center gap-2 text-xs">
          <span className="text-gray-400">SLIPPAGE:</span>
          <input 
            className="w-14 bg-surface-dark border border-white/10 text-center text-white rounded focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary h-6 text-[11px] font-mono" 
            type="text" 
            value={slippage}
            onChange={(e) => setSlippage(e.target.value)}
          />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className="bg-surface-dark border border-white/10 rounded-lg p-3 flex flex-col gap-1">
          <label className="text-[10px] text-gray-400 uppercase font-bold tracking-wider">Position Size (USDC)</label>
          <div className="flex items-center justify-between gap-2">
            <button 
              onClick={onSizeDown}
              className="size-8 flex items-center justify-center rounded bg-white/5 hover:bg-white/10 text-white font-bold transition-colors"
            >
              −
            </button>
            <div className="flex items-center flex-1">
              <span className="text-gray-500 font-mono">$</span>
              <input 
                className="bg-transparent text-white font-mono font-bold text-xl w-full text-right focus:outline-none placeholder-gray-700" 
                placeholder="0.00" 
                type="text" 
                value={formatSize(selectedSize)}
                onChange={handleSizeChange}
              />
            </div>
            <button 
              onClick={onSizeUp}
              className="size-8 flex items-center justify-center rounded bg-white/5 hover:bg-white/10 text-white font-bold transition-colors"
            >
              +
            </button>
          </div>
        </div>
        <div className="flex gap-2">
          {[
            { label: '25%', value: 25 },
            { label: '50%', value: 50 },
            { label: 'MAX', value: 100 },
          ].map((btn) => (
            <button 
              key={btn.label}
              onClick={() => handlePercentage(btn.value)}
              className={`flex-1 rounded-lg text-xs font-bold uppercase transition-all py-2 ${
                btn.label === 'MAX' 
                  ? 'bg-primary/20 border border-primary text-primary shadow-glow-sm hover:bg-primary/30' 
                  : 'bg-surface-dark border border-white/10 hover:border-white/30 text-gray-400 hover:text-white'
              }`}
            >
              {btn.label}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* YES Button */}
        <button 
          onClick={() => handleOrder('YES')}
          disabled={disabled || isSubmitting !== null}
          className={`relative overflow-hidden group h-32 rounded font-bold transition-all flex flex-col items-center justify-center border-4 ring-1 ${
            disabled 
              ? 'bg-gray-800 border-gray-700 text-gray-600 cursor-not-allowed ring-gray-700/50' 
              : isSubmitting === 'YES'
                ? 'bg-primary/80 border-primary text-surface-darker ring-primary/50 animate-pulse'
                : 'bg-primary border-primary text-surface-darker ring-primary/50 shadow-glow'
          }`}
        >
          <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-200"></div>
          <span className="relative z-10 text-xs font-black uppercase tracking-[0.2em] mb-1 opacity-80">
            {isSubmitting === 'YES' ? 'SUBMITTING...' : 'BET YES'}
          </span>
          <div className="relative z-10 flex items-baseline gap-1">
             <span className="text-6xl font-black font-mono tracking-tighter drop-shadow-md">{yesPrice}</span>
             <span className="text-2xl font-bold opacity-60">¢</span>
          </div>
          <span className="relative z-10 text-[10px] font-mono mt-1 font-black opacity-80 tracking-widest">LONG VOL</span>
        </button>
        
        {/* NO Button */}
        <button 
          onClick={() => handleOrder('NO')}
          disabled={disabled || isSubmitting !== null}
          className={`relative overflow-hidden group h-32 rounded font-bold transition-all flex flex-col items-center justify-center border-4 ${
            disabled 
              ? 'bg-gray-800 border-gray-700 text-gray-600 cursor-not-allowed' 
              : isSubmitting === 'NO'
                ? 'bg-accent-red border-accent-red text-white animate-pulse'
                : 'bg-surface-dark border-accent-red text-accent-red hover:bg-accent-red hover:text-white shadow-[0_0_20px_rgba(255,59,48,0.2)]'
          }`}
        >
          <span className="relative z-10 text-xs font-black uppercase tracking-[0.2em] mb-1 opacity-80 group-hover:text-white">
            {isSubmitting === 'NO' ? 'SUBMITTING...' : 'BET NO'}
          </span>
          <div className="relative z-10 flex items-baseline gap-1">
             <span className="text-6xl font-black font-mono tracking-tighter drop-shadow-md">{noPrice}</span>
             <span className="text-2xl font-bold opacity-60 group-hover:text-white/60">¢</span>
          </div>
          <span className="relative z-10 text-[10px] font-mono mt-1 font-black opacity-80 tracking-widest group-hover:text-white/80">SHORT VOL</span>
        </button>
      </div>

      {/* Disabled overlay message */}
      {disabled && (
        <div className="absolute inset-0 bg-black/50 flex items-center justify-center rounded-xl">
          <div className="text-center">
            <span className="text-accent-orange font-bold text-sm uppercase tracking-wider block">
              Trading Disabled
            </span>
            {disabledReason && (
              <span className="text-gray-400 text-xs mt-2 block">
                {disabledReason}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ExecutionControls;
