
import React from 'react';

interface StatCardProps {
  label: string;
  value: string | number;
  subValue?: string;
  subValueColor?: string;
  icon?: string;
  trend?: 'up' | 'down' | 'none';
  highlight?: boolean;
}

const StatCard: React.FC<StatCardProps> = ({ 
  label, value, subValue, subValueColor = 'text-gray-500', icon, trend, highlight 
}) => {
  return (
    <div className={`bg-surface-dark border ${highlight ? 'border-primary/40 shadow-glow-sm' : 'border-white/5'} p-4 rounded-lg relative overflow-hidden group transition-all`}>
      {highlight && <div className="absolute inset-0 bg-primary/5 animate-pulse rounded-lg pointer-events-none" />}
      
      <div className="flex justify-between items-start mb-1">
        <p className={`text-xs font-bold uppercase tracking-wider ${highlight ? 'text-primary' : 'text-gray-400'}`}>
          {label}
        </p>
        {icon && (
          <span className={`material-symbols-outlined text-sm ${highlight ? 'text-primary' : 'text-gray-500 opacity-30 group-hover:opacity-60 transition-opacity'}`}>
            {icon}
          </span>
        )}
      </div>

      <div className="flex items-baseline gap-2">
        <p className={`text-2xl font-bold tracking-tight ${highlight ? 'text-white' : 'text-white'}`}>
          {value}
        </p>
      </div>

      {subValue && (
        <p className={`text-xs mt-1 flex items-center gap-1 font-medium ${subValueColor}`}>
          {trend === 'up' && <span className="material-symbols-outlined text-[14px]">arrow_drop_up</span>}
          {trend === 'down' && <span className="material-symbols-outlined text-[14px]">arrow_drop_down</span>}
          {subValue}
        </p>
      )}
    </div>
  );
};

export default StatCard;
