
import React from 'react';

const marketData = [
  { symbol: 'BTC', name: 'Bitcoin', value: 8.4, volume: '2.3B' },
  { symbol: 'ETH', name: 'Ethereum', value: -3.2, volume: '1.8B' },
  { symbol: 'SOL', name: 'Solana', value: 12.7, volume: '892M' },
  { symbol: 'ADA', name: 'Cardano', value: -1.8, volume: '456M' },
  { symbol: 'DOT', name: 'Polkadot', value: 5.9, volume: '234M' },
  { symbol: 'LINK', name: 'Chainlink', value: -4.3, volume: '345M' },
  { symbol: 'MATIC', name: 'Polygon', value: 7.1, volume: '678M' },
  { symbol: 'AVAX', name: 'Avalanche', value: -2.9, volume: '123M' },
  { symbol: 'UNI', name: 'Uniswap', value: 15.6, volume: '567M' },
  { symbol: 'AAVE', name: 'Aave', value: -7.8, volume: '89M' },
  { symbol: 'CRV', name: 'Curve', value: 3.4, volume: '145M' },
  { symbol: 'MKR', name: 'Maker', value: -5.1, volume: '67M' },
];

const getPerformanceColor = (value: number) => {
  if (value > 10) return 'bg-gradient-to-br from-[var(--accent-primary)] to-[var(--accent-tertiary)]';
  if (value > 5) return 'bg-gradient-to-br from-[var(--positive)] to-[var(--accent-primary)]';
  if (value > 0) return 'bg-gradient-to-br from-[var(--positive-dim)] to-[var(--positive)]';
  if (value > -5) return 'bg-gradient-to-br from-[var(--negative-dim)] to-[var(--negative)]';
  return 'bg-gradient-to-br from-[var(--negative)] to-[#8B0000]';
};

const getIntensity = (value: number) => {
  const absValue = Math.abs(value);
  if (absValue > 10) return 'opacity-100';
  if (absValue > 5) return 'opacity-80';
  if (absValue > 2) return 'opacity-60';
  return 'opacity-40';
};

export const MarketPerformanceGrid = () => {
  return (
    <div className="grid grid-cols-6 gap-3">
      {marketData.map((item) => (
        <div
          key={item.symbol}
          className={`
            ${getPerformanceColor(item.value)} ${getIntensity(item.value)}
            interactive-base rounded-lg p-4 text-center text-white relative overflow-hidden
            hover:scale-105 transition-all duration-300
          `}
          style={{
            boxShadow: item.value > 0 
              ? '0 4px 20px rgba(16, 185, 129, 0.2)' 
              : '0 4px 20px rgba(239, 68, 68, 0.2)'
          }}
        >
          {/* Background pattern */}
          <div 
            className="absolute inset-0 opacity-10"
            style={{
              background: `repeating-linear-gradient(
                45deg,
                transparent,
                transparent 2px,
                rgba(255,255,255,0.1) 2px,
                rgba(255,255,255,0.1) 4px
              )`
            }}
          />
          
          <div className="relative z-10">
            <div className="text-lg font-bold mb-1">{item.symbol}</div>
            <div className="text-xs opacity-80 mb-2">{item.name}</div>
            <div className="text-lg font-mono font-semibold mb-1">
              {item.value > 0 ? '+' : ''}{item.value}%
            </div>
            <div className="text-xs opacity-70">Vol: {item.volume}</div>
          </div>
        </div>
      ))}
    </div>
  );
};