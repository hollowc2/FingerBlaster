import React, { useMemo } from 'react';

interface OrderDepthProps {
  yesAskDepth: number;
  noAskDepth: number;
}

const OrderDepth: React.FC<OrderDepthProps> = ({ yesAskDepth, noAskDepth }) => {
  // Normalize depths to percentages for visualization
  const { bidBars, askBars, totalBid, totalAsk } = useMemo(() => {
    const totalBid = yesAskDepth;
    const totalAsk = noAskDepth;
    const maxDepth = Math.max(totalBid, totalAsk, 1);
    
    // Generate bar heights (simulated distribution for visual appeal)
    const generateBars = (total: number, count: number = 5): number[] => {
      if (total === 0) return Array(count).fill(10);
      
      // Create a realistic distribution
      const weights = [0.35, 0.25, 0.20, 0.12, 0.08];
      return weights.map(w => Math.max(10, (w * total / maxDepth) * 100));
    };
    
    return {
      bidBars: generateBars(totalBid),
      askBars: generateBars(totalAsk),
      totalBid,
      totalAsk,
    };
  }, [yesAskDepth, noAskDepth]);

  // Format volume for display
  const formatVolume = (vol: number): string => {
    if (vol >= 1000) {
      return `${(vol / 1000).toFixed(0)}k`;
    }
    return vol.toFixed(0);
  };

  return (
    <div className="bg-surface-dark border border-white/5 rounded-lg p-4">
      <h3 className="text-[10px] text-gray-400 uppercase tracking-widest mb-3 font-bold">Order Depth</h3>
      <div className="flex gap-1 h-[100px] items-end justify-center">
        {/* Bid Side (YES) */}
        <div className="flex-1 flex flex-col-reverse items-end gap-0.5 h-full">
          {bidBars.map((height, idx) => (
            <div 
              key={`bid-${idx}`}
              className={`w-full rounded-sm transition-all duration-300 ${
                idx === bidBars.length - 1 ? 'border-t border-primary/50' : ''
              }`}
              style={{ 
                height: `${height}%`,
                backgroundColor: `rgba(6, 249, 87, ${0.2 + (idx * 0.15)})`,
              }}
            ></div>
          ))}
        </div>
        
        <div className="w-px h-full bg-white/10 mx-1"></div>
        
        {/* Ask Side (NO) */}
        <div className="flex-1 flex flex-col-reverse items-start gap-0.5 h-full">
          {askBars.map((height, idx) => (
            <div 
              key={`ask-${idx}`}
              className={`w-full rounded-sm transition-all duration-300 ${
                idx === askBars.length - 1 ? 'border-t border-accent-red/50' : ''
              }`}
              style={{ 
                height: `${height}%`,
                backgroundColor: `rgba(255, 59, 48, ${0.2 + (idx * 0.15)})`,
              }}
            ></div>
          ))}
        </div>
      </div>
      <div className="flex justify-between mt-2 text-[10px] text-gray-500 font-mono font-bold">
        <span className="text-primary/80">BID: ${formatVolume(totalBid)}</span>
        <span className="text-accent-red/80">ASK: ${formatVolume(totalAsk)}</span>
      </div>
    </div>
  );
};

export default OrderDepth;
