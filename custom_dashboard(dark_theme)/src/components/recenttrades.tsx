
import React from 'react';
import { Clock, MoreVertical } from 'lucide-react';

const recentTrades = [
  { time: '16:24:32', price: 43685, size: 0.0234, side: 'buy' },
  { time: '16:24:28', price: 43682, size: 0.1567, side: 'sell' },
  { time: '16:24:25', price: 43688, size: 0.0891, side: 'buy' },
  { time: '16:24:22', price: 43680, size: 0.2345, side: 'buy' },
  { time: '16:24:19', price: 43675, size: 0.0456, side: 'sell' },
  { time: '16:24:16', price: 43678, size: 0.1234, side: 'buy' },
  { time: '16:24:13', price: 43672, size: 0.0789, side: 'sell' },
  { time: '16:24:10', price: 43676, size: 0.1890, side: 'buy' },
];

const TradeRow = ({ trade }: { trade: any }) => {
  const isBuy = trade.side === 'buy';
  
  return (
    <div className="flex items-center justify-between py-2 px-2 hover:bg-[var(--bg-hover)] rounded transition-colors">
      <div className="flex items-center space-x-3">
        <div className={`w-2 h-2 rounded-full ${
          isBuy ? 'bg-[var(--positive)]' : 'bg-[var(--negative)]'
        }`} />
        <span className="text-[var(--text-muted)] text-xs text-mono">
          {trade.time}
        </span>
      </div>
      
      <span className={`text-mono font-medium ${
        isBuy ? 'text-[var(--positive)]' : 'text-[var(--negative)]'
      }`}>
        ${trade.price.toLocaleString()}
      </span>
      
      <span className="text-[var(--text-secondary)] text-mono text-sm">
        {trade.size.toFixed(4)}
      </span>
    </div>
  );
};

export const RecentTrades = () => {
  const totalBuyVolume = recentTrades
    .filter(trade => trade.side === 'buy')
    .reduce((sum, trade) => sum + trade.size, 0);
    
  const totalSellVolume = recentTrades
    .filter(trade => trade.side === 'sell')
    .reduce((sum, trade) => sum + trade.size, 0);

  return (
    <div className="panel-base p-4 animate-slide-up">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Clock size={16} className="text-[var(--accent-primary)]" />
          <h2 className="heading-secondary">Recent Trades</h2>
        </div>
        <button className="interactive-base p-1 rounded text-[var(--text-muted)] hover:text-[var(--accent-primary)]">
          <MoreVertical size={16} />
        </button>
      </div>
      
      {/* Header */}
      <div className="flex items-center justify-between text-xs text-[var(--text-muted)] mb-3 px-2">
        <span>Time</span>
        <span>Price (USDT)</span>
        <span>Size (BTC)</span>
      </div>
      
      {/* Trades List */}
      <div className="space-y-1 max-h-64 overflow-y-auto">
        {recentTrades.map((trade, index) => (
          <TradeRow key={index} trade={trade} />
        ))}
      </div>
      
      {/* Summary */}
      <div className="mt-4 pt-3 border-t border-[var(--border-primary)]">
        <div className="grid grid-cols-2 gap-4 text-center">
          <div className="p-2 rounded-lg bg-[var(--positive)] bg-opacity-10">
            <div className="text-[var(--positive)] font-medium text-mono text-sm">
              {totalBuyVolume.toFixed(4)}
            </div>
            <div className="text-[var(--text-muted)] text-xs">Buy Volume</div>
          </div>
          <div className="p-2 rounded-lg bg-[var(--negative)] bg-opacity-10">
            <div className="text-[var(--negative)] font-medium text-mono text-sm">
              {totalSellVolume.toFixed(4)}
            </div>
            <div className="text-[var(--text-muted)] text-xs">Sell Volume</div>
          </div>
        </div>
        
        <div className="text-center mt-3">
          <div className="text-[var(--text-primary)] font-medium text-mono">
            {((totalBuyVolume / (totalBuyVolume + totalSellVolume)) * 100).toFixed(1)}%
          </div>
          <div className="text-[var(--text-muted)] text-xs">Buy Ratio</div>
        </div>
      </div>
    </div>
  );
};
