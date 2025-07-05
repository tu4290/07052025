
import React from 'react';
import { TrendingUp, TrendingDown, MoreHorizontal } from 'lucide-react';

const marketData = [
  { symbol: 'BTC', name: 'Bitcoin', price: 43680, change: 2.34, volume: '2.1B' },
  { symbol: 'ETH', name: 'Ethereum', price: 2580, change: 4.67, volume: '1.8B' },
  { symbol: 'ADA', name: 'Cardano', price: 0.52, change: -1.23, volume: '456M' },
  { symbol: 'SOL', name: 'Solana', price: 98.45, change: 7.89, volume: '892M' },
  { symbol: 'DOT', name: 'Polkadot', price: 7.23, change: -2.45, volume: '234M' },
  { symbol: 'LINK', name: 'Chainlink', price: 14.67, change: 3.21, volume: '345M' },
  { symbol: 'AVAX', name: 'Avalanche', price: 36.78, change: 5.43, volume: '567M' },
  { symbol: 'MATIC', name: 'Polygon', price: 0.89, change: -0.67, volume: '123M' },
];

const HeatmapTile = ({ asset }: { asset: any }) => {
  const isPositive = asset.change >= 0;
  const intensity = Math.min(Math.abs(asset.change) / 10, 1); // Normalize intensity
  
  return (
    <div 
      className={`
        panel-base interactive-base p-4 relative overflow-hidden
        ${isPositive ? 'bg-[var(--positive)]' : 'bg-[var(--negative)]'}
        bg-opacity-10 hover:bg-opacity-20
      `}
      style={{
        backgroundColor: `${isPositive ? 'var(--positive)' : 'var(--negative)'}${Math.floor(intensity * 20).toString(16).padStart(2, '0')}`,
      }}
    >
      <div className="relative z-10">
        <div className="flex items-center justify-between mb-2">
          <span className="font-bold text-[var(--text-primary)]">{asset.symbol}</span>
          <div className={`flex items-center space-x-1 ${
            isPositive ? 'text-[var(--positive)]' : 'text-[var(--negative)]'
          }`}>
            {isPositive ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
            <span className="text-xs font-medium">
              {asset.change >= 0 ? '+' : ''}{asset.change.toFixed(2)}%
            </span>
          </div>
        </div>
        
        <div className="text-[var(--text-secondary)] text-xs mb-1">
          {asset.name}
        </div>
        
        <div className="text-[var(--text-primary)] font-medium text-mono">
          ${asset.price.toLocaleString()}
        </div>
        
        <div className="text-[var(--text-muted)] text-xs mt-1">
          Vol: {asset.volume}
        </div>
      </div>
      
      {/* Subtle pattern overlay */}
      <div className="absolute inset-0 opacity-5 bg-gradient-to-br from-transparent via-white to-transparent" />
    </div>
  );
};

export const MarketHeatmap = () => {
  return (
    <div className="panel-base p-6 animate-slide-up">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="heading-secondary mb-1">Market Overview</h2>
          <p className="text-[var(--text-muted)] text-sm">Top cryptocurrencies by market cap</p>
        </div>
        <button className="interactive-base p-2 rounded-lg text-[var(--text-muted)] hover:text-[var(--accent-primary)]">
          <MoreHorizontal size={16} />
        </button>
      </div>
      
      <div className="grid grid-cols-4 gap-3">
        {marketData.map((asset, index) => (
          <HeatmapTile 
            key={asset.symbol} 
            asset={asset}
          />
        ))}
      </div>
      
      {/* Legend */}
      <div className="flex items-center justify-center space-x-6 mt-6 pt-4 border-t border-[var(--border-primary)]">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded bg-[var(--positive)] opacity-30" />
          <span className="text-[var(--text-muted)] text-sm">Gainers</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded bg-[var(--negative)] opacity-30" />
          <span className="text-[var(--text-muted)] text-sm">Losers</span>
        </div>
        <div className="text-[var(--text-muted)] text-sm">
          Color intensity = Price change magnitude
        </div>
      </div>
    </div>
  );
};
