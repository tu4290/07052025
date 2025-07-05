import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Cell, Tooltip } from 'recharts';

const data = [
  { name: 'BTC', value: 8.4, fullName: 'Bitcoin' },
  { name: 'ETH', value: -3.2, fullName: 'Ethereum' },
  { name: 'SOL', value: 12.7, fullName: 'Solana' },
  { name: 'ADA', value: -1.8, fullName: 'Cardano' },
  { name: 'DOT', value: 5.9, fullName: 'Polkadot' },
  { name: 'LINK', value: -4.3, fullName: 'Chainlink' },
  { name: 'MATIC', value: 7.1, fullName: 'Polygon' },
  { name: 'AVAX', value: -2.9, fullName: 'Avalanche' },
];

const getBarGradient = (value: number) => {
  if (value > 0) {
    if (value > 8) return 'url(#gradientHighPositive)';
    if (value > 4) return 'url(#gradientMediumPositive)';
    return 'url(#gradientLowPositive)';
  } else {
    if (value < -3) return 'url(#gradientHighNegative)';
    return 'url(#gradientLowNegative)';
  }
};

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    const isPositive = data.value > 0;
    
    return (
      <div className="bg-[var(--bg-elevated)] border border-[var(--border-accent)] rounded-lg p-3 shadow-xl backdrop-blur-sm">
        <div className="text-[var(--text-primary)] font-medium mb-1">{data.fullName}</div>
        <div className={`text-mono font-bold text-lg ${isPositive ? 'text-[var(--positive)]' : 'text-[var(--negative)]'}`}>
          {isPositive ? '+' : ''}{data.value}%
        </div>
        <div className="text-[var(--text-muted)] text-xs mt-1">24h Change</div>
      </div>
    );
  }
  return null;
};

export const VerticalBarChart = () => {
  return (
    <div className="h-80 relative">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <defs>
            {/* Positive Gradients */}
            <linearGradient id="gradientHighPositive" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--accent-primary)" stopOpacity={0.9} />
              <stop offset="100%" stopColor="var(--accent-primary)" stopOpacity={0.3} />
            </linearGradient>
            <linearGradient id="gradientMediumPositive" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--positive)" stopOpacity={0.8} />
              <stop offset="100%" stopColor="var(--positive)" stopOpacity={0.2} />
            </linearGradient>
            <linearGradient id="gradientLowPositive" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--positive-dim)" stopOpacity={0.7} />
              <stop offset="100%" stopColor="var(--positive-dim)" stopOpacity={0.1} />
            </linearGradient>
            
            {/* Negative Gradients */}
            <linearGradient id="gradientHighNegative" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--negative)" stopOpacity={0.8} />
              <stop offset="100%" stopColor="var(--negative)" stopOpacity={0.2} />
            </linearGradient>
            <linearGradient id="gradientLowNegative" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--negative-dim)" stopOpacity={0.7} />
              <stop offset="100%" stopColor="var(--negative-dim)" stopOpacity={0.1} />
            </linearGradient>
            
            {/* Bar Shadow Filter */}
            <filter id="barShadow" x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="2" stdDeviation="3" floodColor="rgba(0,0,0,0.3)" />
            </filter>
          </defs>
          
          <CartesianGrid 
            strokeDasharray="2 4" 
            stroke="var(--border-primary)" 
            strokeOpacity={0.3}
            vertical={false}
          />
          <XAxis 
            dataKey="name" 
            axisLine={false}
            tickLine={false}
            tick={{ 
              fill: 'var(--text-muted)', 
              fontSize: 12,
              fontWeight: 500
            }}
          />
          <YAxis 
            axisLine={false}
            tickLine={false}
            tick={{ 
              fill: 'var(--text-muted)', 
              fontSize: 12,
              fontFamily: 'JetBrains Mono, monospace'
            }}
            domain={['dataMin - 2', 'dataMax + 2']}
          />
          <Tooltip content={<CustomTooltip />} />
          <Bar 
            dataKey="value" 
            radius={[6, 6, 2, 2]}
            filter="url(#barShadow)"
            style={{
              filter: 'drop-shadow(0 4px 8px rgba(0,0,0,0.15))'
            }}
          >
            {data.map((entry, index) => (
              <Cell 
                key={`cell-${index}`} 
                fill={getBarGradient(entry.value)}
                className="hover:brightness-110 transition-all duration-300 ease-out"
                style={{
                  filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.2))'
                }}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};